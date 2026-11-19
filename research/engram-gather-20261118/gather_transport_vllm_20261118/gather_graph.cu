// Private prototype: one reader, one CPU service worker, legacy peer-memory MR.
#include <cuda.h>
#include <cuda_runtime.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <semaphore.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cerrno>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "../mechanism/with_indices/verbs_probe.cpp"

static thread_local std::string gg_error;
static void ck(cudaError_t x,const char *what){if(x!=cudaSuccess)throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(x));}
static int gg_fail(const std::exception &e){gg_error=e.what();return -1;}
extern "C" const char *gg_last_error(){return gg_error.c_str();}
static void io(int fd,void *p,size_t n,bool send_data){char *b=(char*)p;while(n){ssize_t z=send_data?send(fd,b,n,MSG_NOSIGNAL):recv(fd,b,n,0);if(z<=0)throw std::runtime_error("gather socket closed/error");b+=z;n-=z;}}
static void timeout_socket(int fd){timeval tv{8,0};setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&tv,sizeof(tv));setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,&tv,sizeof(tv));}
struct Ctrl {unsigned long long seq,ack,count,table,payload,src_rows,error,worker_error;unsigned long long indices[1];};
struct Request {uint64_t kind,seq,count,table,payload,src_rows;};
struct Response {uint64_t seq,error;};
struct PeerMeta {Meta data;uint64_t flag_addr,flag_rkey,arena_bytes;};
struct Owner {int gpu,listener=-1;std::atomic<int> client{-1},stopping{0};Endpoint *ep=nullptr;ibv_mr *scratch_mr=nullptr;void *scratch=nullptr;std::vector<ibv_mr*> tables;std::vector<uint64_t> addresses,lengths;std::thread thread;std::atomic<int> failed{0},connected{0};};
struct Reader {int gpu,fd=-1;Endpoint *ep=nullptr;uint64_t arena_bytes=0;unsigned qps=0,maxn=0;void *device_flags=nullptr;ibv_mr *flags_mr=nullptr;Ctrl *host=nullptr,*dev=nullptr;sem_t sem{};std::thread worker;std::atomic<int> stop{0},failed{0},connected{0};std::mutex lifecycle;std::condition_variable execution_changed;bool closing=false,quiesced=false;int execution_state=0;cudaEvent_t consumed=nullptr;cudaStream_t stream=nullptr;bool stream_set=false,sem_ready=false;};
static void owner_setup_cleanup(Owner *o){if(!o)return;if(o->listener>=0)close(o->listener);if(o->scratch_mr)ibv_dereg_mr(o->scratch_mr);if(o->ep){for(size_t i=1;i<o->tables.size();i++)ibv_dereg_mr(o->tables[i]);probe_close(o->ep);}if(o->scratch)cudaFreeHost(o->scratch);delete o;}
static void reader_setup_cleanup(Reader *r){if(!r)return;if(r->fd>=0)close(r->fd);if(r->sem_ready)sem_destroy(&r->sem);if(r->host)cudaFreeHost(r->host);if(r->consumed)cudaEventDestroy(r->consumed);if(r->flags_mr)ibv_dereg_mr(r->flags_mr);if(r->device_flags)cudaFree(r->device_flags);if(r->ep)probe_close(r->ep);delete r;}
static void local_meta(Endpoint *ep,Meta &m){if(probe_meta(ep,&m))throw std::runtime_error(probe_error());}
static void owner_flags(Owner *o,const PeerMeta &r,uint64_t seq){
 auto &e=*o->ep;uint64_t *scratch=(uint64_t*)o->scratch;scratch[e.qps.size()]=seq;
 for(size_t q=0;q<e.qps.size();q++){ibv_sge s{};s.addr=(uint64_t)(scratch+q);s.length=8;s.lkey=o->scratch_mr->lkey;ibv_send_wr w{};w.sg_list=&s;w.num_sge=1;w.opcode=IBV_WR_RDMA_READ;w.send_flags=IBV_SEND_SIGNALED;w.wr.rdma.remote_addr=r.flag_addr+16*8;w.wr.rdma.rkey=r.flag_rkey;post(e,q,w);}poll(e,e.qps.size());
 for(size_t q=0;q<e.qps.size();q++){ibv_sge s{};s.addr=(uint64_t)(scratch+e.qps.size());s.length=8;s.lkey=o->scratch_mr->lkey;ibv_send_wr w{};w.sg_list=&s;w.num_sge=1;w.opcode=IBV_WR_RDMA_WRITE;w.send_flags=IBV_SEND_SIGNALED|IBV_SEND_FENCE|IBV_SEND_INLINE;w.wr.rdma.remote_addr=r.flag_addr+q*8;w.wr.rdma.rkey=r.flag_rkey;post(e,q,w);}poll(e,e.qps.size());
}
static void owner_thread(Owner *o){
 try {sockaddr_in from{};socklen_t len=sizeof(from);o->client=accept(o->listener,(sockaddr*)&from,&len);if(o->client<0){if(o->stopping)return;throw std::runtime_error("owner accept failed");}close(o->listener);o->listener=-1;PeerMeta peer{},me{};io(o->client,&peer,sizeof(peer),false);local_meta(o->ep,me.data);io(o->client,&me,sizeof(me),true);if(probe_connect(o->ep,&peer.data))throw std::runtime_error(probe_error());o->connected=1;std::vector<uint64_t> ids;
  for(;;){Request req{};io(o->client,&req,sizeof(req),false);if(req.kind==2){Response res{req.seq,0};io(o->client,&res,sizeof(res),true);break;}Response res{req.seq,0};
   try {if(req.kind!=1||req.table>=o->tables.size()||req.count>1048576||req.src_rows>o->lengths[req.table]/264||req.payload>peer.arena_bytes||req.count*264>peer.arena_bytes-req.payload)throw std::runtime_error("invalid request descriptor");
    ids.resize(req.count);if(req.count)io(o->client,ids.data(),req.count*8,false);for(auto id:ids)if(id>=req.src_rows)throw std::runtime_error("remote row out of bounds");
    o->ep->mr=o->tables[req.table];o->ep->addr=o->addresses[req.table];o->ep->bytes=o->lengths[req.table];Meta dest=peer.data;dest.addr+=req.payload;
#ifdef GG_TEST_BAD_RKEY
    dest.rkey ^= 0x80000000u;
#endif
    if(req.count&&probe_data(o->ep,&dest,ids.data(),req.count,1,15,16))throw std::runtime_error(probe_error());
    owner_flags(o,peer,req.seq);
   } catch(...) {res.error=1;o->failed=1;try{io(o->client,&res,sizeof(res),true);}catch(...){}return;}
   io(o->client,&res,sizeof(res),true);
  }
 }catch(...){o->failed=1;}
}
__device__ unsigned long long now_ns(){unsigned long long x;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(x));return x;}
__global__ void stage(Ctrl *c,const long long *ids,unsigned n,unsigned table,uint64_t payload,uint64_t lo,uint64_t hi,uint64_t rows,const long long *reps,const long long *work,const unsigned long long *work_count,long long *dense,long long *inverse,unsigned long long *stats){
 __shared__ unsigned long long cnt;__shared__ int bad,poisoned;
 if(threadIdx.x==0){poisoned=(*(volatile unsigned long long*)&c->error)!=0||(*(volatile unsigned long long*)&c->worker_error)!=0;bad=0;cnt=poisoned?0:(work_count?*work_count:n);if(cnt>n)bad=1;c->seq++;if(!poisoned){c->count=cnt;c->table=table;c->payload=payload;c->src_rows=rows;}}
 __syncthreads();
 for(int j=threadIdx.x;j<16*8;j+=blockDim.x)stats[j]=0;
 for(unsigned q=threadIdx.x;!poisoned&&q<n;q+=blockDim.x){if(inverse)inverse[q]=-1;long long id=ids[q];if(id<0||(uint64_t)id>=rows)atomicExch(&bad,1);}
 __syncthreads();
 for(unsigned long long u=threadIdx.x;u<cnt&&u<n;u+=blockDim.x){long long qi=work?work[u]:(long long)u;if(qi<0||(unsigned long long)qi>=n){atomicExch(&bad,1);continue;}long long id=ids[qi];if(id<0||(uint64_t)id<lo||(uint64_t)id>=hi||(uint64_t)id>=rows){atomicExch(&bad,1);continue;}c->indices[u]=id;if(inverse)inverse[qi]=u;}
 __syncthreads();
 if(!poisoned&&reps&&dense&&inverse)for(unsigned q=threadIdx.x;q<n;q+=blockDim.x){long long r=reps[q];if(r==-1)dense[q]=-1;else if(r<0||(uint64_t)r>=n){atomicExch(&bad,1);dense[q]=-1;}else {long long d=inverse[r];if(d<0||(uint64_t)d>=cnt){atomicExch(&bad,1);dense[q]=-1;}else dense[q]=d;}}
 __threadfence_system(); // Every writer publishes its mapped indices.
 __syncthreads();
 if(threadIdx.x==0){if(bad){c->count=0;c->error=2;}__threadfence_system();}
}
__global__ void wait_result(Ctrl *c,volatile unsigned long long *flags,unsigned qps,unsigned long long *stats){
 if(threadIdx.x) return;volatile Ctrl *v=c;uint64_t seq=v->seq, deadline=now_ns()+8000000000ull;bool done=false;
 while(now_ns()<deadline){if(v->ack==seq){__threadfence_system();if(v->error||v->worker_error||v->count==0){done=true;break;}bool all=true;for(unsigned q=0;q<qps;q++)all&=flags[q]==seq;if(all){done=true;break;}}}
 uint64_t error=v->error;if(!error)error=v->worker_error;if(!done&&!error){c->error=3;__threadfence_system();error=3;}
 if(error){for(unsigned q=0;q<qps;q++)stats[q*8]=error;}else {uint64_t n=v->count;for(unsigned q=0;q<qps;q++)stats[q*8+4]=(n*(q+1)/qps)-(n*q/qps);}
}
static void CUDART_CB wake(void *p){sem_post((sem_t*)p);}
static void reader_worker(Reader *r){
 for(;;){while(sem_wait(&r->sem)&&errno==EINTR){}if(r->stop)break;auto *c=r->host;uint64_t seq=__atomic_load_n(&c->seq,__ATOMIC_ACQUIRE);uint64_t err=__atomic_load_n(&c->error,__ATOMIC_ACQUIRE);if(!err)err=__atomic_load_n(&c->worker_error,__ATOMIC_ACQUIRE);
  if(!err && c->count){try{Request req{1,seq,c->count,c->table,c->payload,c->src_rows};io(r->fd,&req,sizeof(req),true);io(r->fd,c->indices,c->count*8,true);Response res{};io(r->fd,&res,sizeof(res),false);if(res.seq!=seq||res.error)throw std::runtime_error("owner failed");}catch(...){err=1;r->failed=1;}}
  if(err)__atomic_store_n(&c->worker_error,err,__ATOMIC_RELEASE);__atomic_store_n(&c->ack,seq,__ATOMIC_RELEASE);
 }
}
extern "C" void *gg_owner_create(int gpu,const char *device,int gid,unsigned qps,const uint64_t *addr,const uint64_t *lens,unsigned tables,unsigned port){
 Owner *o=nullptr;try{if(!device||!addr||!lens||!tables||!addr[0]||!lens[0]||lens[0]%264||tables>1024||qps!=16||!port||port>65535)throw std::runtime_error("invalid owner arguments");ck(cudaSetDevice(gpu),"owner CUDA device");o=new Owner;o->gpu=gpu;o->tables.reserve(tables);o->addresses.reserve(tables);o->lengths.reserve(tables);
  o->ep=(Endpoint*)probe_create(device,gid,qps,addr[0],lens[0]);if(!o->ep)throw std::runtime_error(probe_error());o->tables.push_back(o->ep->mr);o->addresses.push_back(addr[0]);o->lengths.push_back(lens[0]);
  for(unsigned i=1;i<tables;i++){if(!addr[i]||!lens[i]||lens[i]%264)throw std::runtime_error("invalid table");auto *mr=ibv_reg_mr(o->ep->pd,(void*)addr[i],lens[i],IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_READ|IBV_ACCESS_REMOTE_WRITE);if(!mr)throw std::runtime_error("host table registration failed");o->tables.push_back(mr);o->addresses.push_back(addr[i]);o->lengths.push_back(lens[i]);}
  ck(cudaHostAlloc(&o->scratch,(qps+1)*8,cudaHostAllocDefault),"owner host scratch");o->scratch_mr=ibv_reg_mr(o->ep->pd,o->scratch,(qps+1)*8,IBV_ACCESS_LOCAL_WRITE);if(!o->scratch_mr)throw std::runtime_error("owner scratch MR failed");
  o->listener=socket(AF_INET,SOCK_STREAM,0);if(o->listener<0)throw std::runtime_error("owner socket");int yes=1;setsockopt(o->listener,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof(yes));sockaddr_in a{};a.sin_family=AF_INET;a.sin_addr.s_addr=INADDR_ANY;a.sin_port=htons(port);if(bind(o->listener,(sockaddr*)&a,sizeof(a))||listen(o->listener,1))throw std::runtime_error("owner bind/listen");o->thread=std::thread(owner_thread,o);return o;
 }catch(const std::exception&e){owner_setup_cleanup(o);gg_fail(e);return nullptr;}}
extern "C" void *gg_reader_create(int gpu,const char *device,int gid,unsigned qps,uint64_t arena,uint64_t bytes,unsigned maxn){
 Reader *r=nullptr;try{if(!device||qps!=16||!arena||!bytes||!maxn||maxn>1048576)throw std::runtime_error("invalid reader arguments");ck(cudaSetDevice(gpu),"reader CUDA device");int ordering=0;CUdevice d;if(cuDeviceGet(&d,gpu)!=CUDA_SUCCESS||cuDeviceGetAttribute(&ordering,CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING,d)!=CUDA_SUCCESS||ordering<100)throw std::runtime_error("GPU Direct RDMA OWNER ordering unavailable");r=new Reader;r->gpu=gpu;r->arena_bytes=bytes;r->qps=qps;r->maxn=maxn;ck(cudaEventCreateWithFlags(&r->consumed,cudaEventDisableTiming),"reader completion event");
  r->ep=(Endpoint*)probe_create(device,gid,qps,arena,bytes);if(!r->ep)throw std::runtime_error(probe_error());ck(cudaMalloc(&r->device_flags,17*8),"flags cudaMalloc");ck(cudaMemset(r->device_flags,0,17*8),"flags clear");r->flags_mr=ibv_reg_mr(r->ep->pd,r->device_flags,17*8,IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_READ|IBV_ACCESS_REMOTE_WRITE);if(!r->flags_mr)throw std::runtime_error("legacy GPU flags MR failed");
  size_t size=offsetof(Ctrl,indices)+size_t(maxn)*8;ck(cudaHostAlloc((void**)&r->host,size,cudaHostAllocMapped),"mapped control allocation");memset(r->host,0,size);ck(cudaHostGetDevicePointer((void**)&r->dev,r->host,0),"mapped control pointer");if(sem_init(&r->sem,0,0))throw std::runtime_error("sem_init failed");r->sem_ready=true;return r;
 }catch(const std::exception&e){reader_setup_cleanup(r);gg_fail(e);return nullptr;}}
extern "C" int gg_reader_connect(void *ptr,const char *host,unsigned port){bool started=false;try{auto *r=(Reader*)ptr;if(!r||!host||!port||port>65535||r->connected)throw std::runtime_error("invalid connect");started=true;r->fd=socket(AF_INET,SOCK_STREAM,0);sockaddr_in a{};a.sin_family=AF_INET;a.sin_port=htons(port);if(inet_pton(AF_INET,host,&a.sin_addr)!=1||connect(r->fd,(sockaddr*)&a,sizeof(a)))throw std::runtime_error("reader connect");timeout_socket(r->fd);PeerMeta me{},peer{};local_meta(r->ep,me.data);me.flag_addr=(uint64_t)r->device_flags;me.flag_rkey=r->flags_mr->rkey;me.arena_bytes=r->arena_bytes;io(r->fd,&me,sizeof(me),true);io(r->fd,&peer,sizeof(peer),false);if(probe_connect(r->ep,&peer.data))throw std::runtime_error(probe_error());r->worker=std::thread(reader_worker,r);r->connected=1;return 0;}catch(const std::exception&e){if(started&&ptr)((Reader*)ptr)->failed=1;return gg_fail(e);}}
extern "C" int gg_fetch(void *ptr,uint64_t ids,unsigned n,uint64_t payload,unsigned table,uint64_t lo,uint64_t hi,uint64_t rows,uint64_t reps,uint64_t work,uint64_t work_count,uint64_t dense,uint64_t inverse,uint64_t stats,uint64_t stream){bool submitted=false;try{auto *r=(Reader*)ptr;if(!r)throw std::runtime_error("invalid reader");std::lock_guard<std::mutex> guard(r->lifecycle);if(r->closing||r->execution_state!=1||!r->connected||r->failed||!ids||!stats||n>r->maxn||lo>hi||payload>r->arena_bytes||uint64_t(n)*264>r->arena_bytes-payload)throw std::runtime_error("invalid fetch or poisoned reader");if((work||work_count||reps||dense||inverse)&&!(work&&work_count&&reps&&dense&&inverse))throw std::runtime_error("mapping pointers must be supplied together");auto s=(cudaStream_t)stream;if(r->stream_set&&r->stream!=s)throw std::runtime_error("one stream per reader");r->stream=s;r->stream_set=true;
  submitted=true;stage<<<1,128,0,s>>>(r->dev,(long long*)ids,n,table,payload,lo,hi,rows,(long long*)reps,(long long*)work,(unsigned long long*)work_count,(long long*)dense,(long long*)inverse,(unsigned long long*)stats);ck(cudaGetLastError(),"stage launch");ck(cudaLaunchHostFunc(s,wake,&r->sem),"host callback");wait_result<<<1,32,0,s>>>(r->dev,(unsigned long long*)r->device_flags,r->qps,(unsigned long long*)stats);ck(cudaGetLastError(),"wait launch");return 0;}catch(const std::exception&e){if(submitted&&ptr)((Reader*)ptr)->failed=1;return gg_fail(e);}}
extern "C" int gg_record_consumed(void *ptr,uint64_t stream){auto *r=(Reader*)ptr;if(!r||!r->stream_set||r->stream!=(cudaStream_t)stream){gg_error="invalid consumed stream";return -1;}return 0;}
// Lease the entire execution, including input/workspace writes and graph replay.
// Application launch paths must call begin before touching shared inputs, then
// end on a stream ordered after the last consumer. A caller launching a graph
// without this protocol is outside this ABI's safety contract.
static int begin_execution(void *ptr,unsigned timeout_ms){
 try {auto *r=(Reader*)ptr;if(!r)throw std::runtime_error("invalid reader");
  std::unique_lock<std::mutex> guard(r->lifecycle);
  ck(cudaSetDevice(r->gpu),"begin execution device");
  auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(timeout_ms);
  for(;;){
   if(r->closing||r->failed||!r->connected)throw std::runtime_error("reader unavailable");
   bool ready=r->execution_state==0;
   if(r->execution_state==2){
    auto status=cudaEventQuery(r->consumed);
    if(status==cudaSuccess)ready=true;
    else if(status!=cudaErrorNotReady){r->failed=1;ck(status,"query reader consumer");}
   }
   if(ready){
    if(__atomic_load_n(&r->host->error,__ATOMIC_ACQUIRE)||
       __atomic_load_n(&r->host->worker_error,__ATOMIC_ACQUIRE)){
     r->failed=1;throw std::runtime_error("poisoned reader");
    }
    r->execution_state=1;return 0;
   }
   auto now=std::chrono::steady_clock::now();
   if(!timeout_ms||now>=deadline)throw std::runtime_error("reader execution busy");
   auto next=now+std::chrono::milliseconds(1);
   r->execution_changed.wait_until(guard,next<deadline?next:deadline);
  }
 }catch(const std::exception&e){return gg_fail(e);}
}
extern "C" int gg_reader_begin_execution(void *ptr){return begin_execution(ptr,0);}
extern "C" int gg_reader_begin_execution_wait(void *ptr,unsigned timeout_ms){
 if(!timeout_ms){gg_error="wait timeout must be positive";return -1;}
 return begin_execution(ptr,timeout_ms);
}
extern "C" int gg_reader_end_execution(void *ptr,uint64_t stream){
 try {auto *r=(Reader*)ptr;if(!r)throw std::runtime_error("invalid reader");
  std::lock_guard<std::mutex> guard(r->lifecycle);
  if(r->closing||r->failed||r->execution_state!=1)throw std::runtime_error("reader execution is not leased");
  try {ck(cudaSetDevice(r->gpu),"end execution device");
   ck(cudaEventRecord(r->consumed,(cudaStream_t)stream),"record final consumer");
   r->execution_state=2;r->execution_changed.notify_all();
  }catch(...){r->failed=1;r->execution_changed.notify_all();throw;}
  return 0;
 }catch(const std::exception&e){return gg_fail(e);}
}
// The caller must first prevent every existing graph from being launched again.
// A successful prepare has retired GPU/NIC work and the worker, but retains MRs,
// CUDA allocations, and the reader. Graph owners can then destroy their graphs.
extern "C" int gg_reader_prepare_close(void *ptr){
 try {auto *r=(Reader*)ptr;if(!r)throw std::runtime_error("invalid reader");
  std::lock_guard<std::mutex> guard(r->lifecycle);
  if(r->quiesced)return 0;
  if(r->execution_state==1)throw std::runtime_error("reader execution still leased");
  if(r->closing)throw std::runtime_error("reader retirement previously failed");
  r->closing=true;r->execution_changed.notify_all();
  try {
   ck(cudaSetDevice(r->gpu),"prepare close device");
   ck(cudaDeviceSynchronize(),"prepare close synchronize");
   if(r->failed||__atomic_load_n(&r->host->error,__ATOMIC_ACQUIRE)||
      __atomic_load_n(&r->host->worker_error,__ATOMIC_ACQUIRE))
    throw std::runtime_error("poisoned: all reader resources retained");
   if(r->connected){
    Request req{2,0,0,0,0,0};io(r->fd,&req,sizeof(req),true);
    Response res{};io(r->fd,&res,sizeof(res),false);
    if(res.error)throw std::runtime_error("owner retirement rejected");
    r->stop=1;
    if(sem_post(&r->sem))throw std::runtime_error("worker retirement wake failed");
    if(r->worker.joinable())r->worker.join();
    if(close(r->fd))throw std::runtime_error("reader socket close failed");
    r->fd=-1;r->connected=0;
   }
   r->quiesced=true;
  }catch(...){r->failed=1;throw;}
  return 0;
 }catch(const std::exception&e){return gg_fail(e);}
}
extern "C" int gg_reader_close(void *ptr){
 try {auto *r=(Reader*)ptr;if(!r)throw std::runtime_error("invalid reader");
  std::unique_lock<std::mutex> guard(r->lifecycle);
  if(!r->quiesced||r->failed)throw std::runtime_error("reader must be quiesced before close");
  if(ibv_dereg_mr(r->flags_mr))throw std::runtime_error("flag MR deregistration failed");
  r->flags_mr=nullptr;
  probe_close(r->ep);r->ep=nullptr;
  ck(cudaFree(r->device_flags),"free reader flags");r->device_flags=nullptr;
  ck(cudaFreeHost(r->host),"free reader control");r->host=nullptr;
  ck(cudaEventDestroy(r->consumed),"destroy reader consumer event");r->consumed=nullptr;
  if(sem_destroy(&r->sem))throw std::runtime_error("reader semaphore destroy failed");
  guard.unlock();delete r;return 0;
 }catch(const std::exception&e){if(ptr)((Reader*)ptr)->failed=1;return gg_fail(e);}
}
extern "C" int gg_owner_close(void *ptr){try{auto *o=(Owner*)ptr;if(!o)throw std::runtime_error("invalid owner");if(o->failed)throw std::runtime_error("poisoned owner: registrations retained until process exit");if(o->client<0){o->stopping=1;if(o->listener>=0)shutdown(o->listener,SHUT_RDWR);}if(o->thread.joinable())o->thread.join();if(o->failed)throw std::runtime_error("owner failed: registrations retained until process exit");if(o->client>=0)close(o->client);if(o->listener>=0)close(o->listener);ibv_dereg_mr(o->scratch_mr);o->ep->mr=o->tables[0];for(size_t i=1;i<o->tables.size();i++)ibv_dereg_mr(o->tables[i]);probe_close(o->ep);cudaFreeHost(o->scratch);delete o;return 0;}catch(const std::exception&e){return gg_fail(e);}}
