// Opt-in H20 verbs mechanism probe. No CUDA headers or DMA-BUF registration.
#include <infiniband/verbs.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <time.h>

static thread_local std::string error_text;
struct Endpoint {
 ibv_context *ctx{}; ibv_pd *pd{}; ibv_cq *cq{}; ibv_mr *mr{};
 std::vector<ibv_qp*> qps; ibv_port_attr pa{}; ibv_gid gid{};
 uint64_t addr{}; size_t bytes{}; int gid_index{};
};
struct Meta {uint64_t addr; uint32_t rkey; uint32_t mtu; uint32_t lid; uint8_t gid[16]; uint32_t qpn[16];};
static void fail(const char* m) { throw std::runtime_error(m); }
template<typename F> static int guard(F f) { try { f(); return 0; } catch(const std::exception& e){error_text=e.what(); return -1;} }
extern "C" const char *probe_error() {return error_text.c_str();}
extern "C" void *probe_create(const char *name,int gid_index,int nqp,uint64_t addr,size_t bytes) {
 error_text.clear(); Endpoint* e=nullptr; int rc=guard([&]{
  if(nqp<1||nqp>16||!addr||!bytes) fail("invalid create args");
  int n=0; auto **list=ibv_get_device_list(&n); if(!list)fail("ibv_get_device_list");
  ibv_device *found=nullptr; for(int i=0;i<n;i++) if(!strcmp(ibv_get_device_name(list[i]),name))found=list[i];
  if(!found){ibv_free_device_list(list);fail("device not found");}
  auto *p=new Endpoint; e=p; p->ctx=ibv_open_device(found); ibv_free_device_list(list);
  if(!p->ctx)fail("ibv_open_device");
  if(ibv_query_port(p->ctx,1,&p->pa)||p->pa.state!=IBV_PORT_ACTIVE) fail("ibv_query_port/inactive");
  if(ibv_query_gid(p->ctx,1,gid_index,&p->gid))fail("ibv_query_gid"); p->gid_index=gid_index;
  p->pd=ibv_alloc_pd(p->ctx); if(!p->pd)fail("ibv_alloc_pd");
  p->cq=ibv_create_cq(p->ctx,nqp*64,nullptr,nullptr,0); if(!p->cq)fail("ibv_create_cq");
  p->addr=addr;p->bytes=bytes;
  p->mr=ibv_reg_mr(p->pd,reinterpret_cast<void*>(addr),bytes,IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_READ|IBV_ACCESS_REMOTE_WRITE);
  if(!p->mr)fail("ibv_reg_mr legacy pointer registration failed");
  for(int i=0;i<nqp;i++) {
   ibv_qp_init_attr q{};q.send_cq=p->cq;q.recv_cq=p->cq;q.qp_type=IBV_QPT_RC;
   q.cap.max_send_wr=32;q.cap.max_recv_wr=1;q.cap.max_send_sge=15;q.cap.max_recv_sge=1;q.cap.max_inline_data=8;
   auto *qp=ibv_create_qp(p->pd,&q);if(!qp)fail("ibv_create_qp");p->qps.push_back(qp);
   if(q.cap.max_send_sge<15 || q.cap.max_send_wr<16 || q.cap.max_inline_data<8) fail("QP lacks required SGE/credit capability");
   ibv_qp_attr a{};a.qp_state=IBV_QPS_INIT;a.pkey_index=0;a.port_num=1;a.qp_access_flags=IBV_ACCESS_REMOTE_READ|IBV_ACCESS_REMOTE_WRITE;
   if(ibv_modify_qp(qp,&a,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS))fail("QP INIT");
  }
 }); if(rc) { if(e) { for(auto q:e->qps) ibv_destroy_qp(q); if(e->mr)ibv_dereg_mr(e->mr); if(e->cq)ibv_destroy_cq(e->cq); if(e->pd)ibv_dealloc_pd(e->pd); if(e->ctx)ibv_close_device(e->ctx); delete e; } return nullptr; } return e;
}
extern "C" int probe_meta(void *h,Meta *m){return guard([&]{auto &e=*static_cast<Endpoint*>(h);memset(m,0,sizeof(*m));m->addr=e.addr;m->rkey=e.mr->rkey;m->mtu=e.pa.active_mtu;m->lid=e.pa.lid;memcpy(m->gid,&e.gid,16);for(size_t i=0;i<e.qps.size();i++)m->qpn[i]=e.qps[i]->qp_num;});}
extern "C" int probe_connect(void *h,const Meta *m){return guard([&]{auto &e=*static_cast<Endpoint*>(h);
 for(size_t i=0;i<e.qps.size();i++) {ibv_qp_attr a{};a.qp_state=IBV_QPS_RTR;a.path_mtu=static_cast<ibv_mtu>(std::min<unsigned>(e.pa.active_mtu,m->mtu));a.dest_qp_num=m->qpn[i];a.rq_psn=0;a.max_dest_rd_atomic=16;a.min_rnr_timer=12;
  a.ah_attr.is_global=1;memcpy(&a.ah_attr.grh.dgid,m->gid,16);a.ah_attr.grh.sgid_index=e.gid_index;a.ah_attr.grh.hop_limit=64;a.ah_attr.dlid=m->lid;a.ah_attr.port_num=1;
  if(ibv_modify_qp(e.qps[i],&a,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN|IBV_QP_MAX_DEST_RD_ATOMIC|IBV_QP_MIN_RNR_TIMER))fail("QP RTR");
  a={};a.qp_state=IBV_QPS_RTS;a.timeout=18;a.retry_cnt=7;a.rnr_retry=7;a.sq_psn=0;a.max_rd_atomic=16;
  if(ibv_modify_qp(e.qps[i],&a,IBV_QP_STATE|IBV_QP_TIMEOUT|IBV_QP_RETRY_CNT|IBV_QP_RNR_RETRY|IBV_QP_SQ_PSN|IBV_QP_MAX_QP_RD_ATOMIC))fail("QP RTS");
 }});}
static void post(Endpoint &e,int q,ibv_send_wr &w){ibv_send_wr *bad=nullptr;int rc=ibv_post_send(e.qps[q],&w,&bad);if(rc)fail("ibv_post_send failed");}
static void poll(Endpoint &e,int count){ibv_wc wc[64];timespec t;clock_gettime(CLOCK_MONOTONIC,&t);uint64_t deadline=uint64_t(t.tv_sec)*1000000000ull+t.tv_nsec+30000000000ull;
 while(count){int n=ibv_poll_cq(e.cq,std::min(count,64),wc);if(n<0)fail("ibv_poll_cq failed");for(int i=0;i<n;i++)if(wc[i].status!=IBV_WC_SUCCESS){error_text=std::string("CQ status: ")+ibv_wc_status_str(wc[i].status);throw std::runtime_error(error_text);}count-=n;
 clock_gettime(CLOCK_MONOTONIC,&t);if(uint64_t(t.tv_sec)*1000000000ull+t.tv_nsec>deadline)fail("CQ timeout");}}
// A single thread posts batches <=16 per QP, then drains every completion.
// Mode 0 READ: local destination is GPU and remote is host table.
// Mode 1 WRITE: local source is host table and remote is GPU destination.
extern "C" int probe_data(void *h,const Meta *remote,const uint64_t *ids,uint32_t count,int mode,int sges,int credits){return guard([&]{
 auto &e=*static_cast<Endpoint*>(h);if(mode<0||mode>1||sges<1||sges>15||credits<1||credits>16)fail("bad transfer args");
 const size_t nq=e.qps.size(); std::vector<uint32_t> next(nq),end(nq);for(size_t q=0;q<nq;q++){next[q]=uint64_t(count)*q/nq;end[q]=uint64_t(count)*(q+1)/nq;}
 bool work=true;while(work){work=false;int posted=0;
 for(size_t q=0;q<nq;q++) {
  ibv_send_wr wr[16]{}; ibv_sge sg[16][15]{};int nw=0;
  while(nw<credits && next[q]<end[q]) {
   work=true;uint32_t pos=next[q];int batch=mode?std::min<int>(sges,end[q]-pos):1;
   for(int j=0;j<batch;j++){sg[nw][j].addr=mode?e.addr+uint64_t(ids[pos+j])*264:e.addr+uint64_t(pos)*264;sg[nw][j].length=264;sg[nw][j].lkey=e.mr->lkey;}
   auto &w=wr[nw];w.wr_id=q;w.sg_list=sg[nw];w.num_sge=batch;w.opcode=mode?IBV_WR_RDMA_WRITE:IBV_WR_RDMA_READ;w.send_flags=IBV_SEND_SIGNALED;
   w.wr.rdma.remote_addr=mode?remote->addr+uint64_t(pos)*264:remote->addr+uint64_t(ids[pos])*264;w.wr.rdma.rkey=remote->rkey;
   if(nw)wr[nw-1].next=&w;next[q]+=batch;nw++;
  }
  if(nw){post(e,q,wr[0]);posted+=nw;}
 }
 if(posted)poll(e,posted);}
 });}
// Owner fences previous per-QP writes by an RDMA READ, then writes a sequence
// flag with IBV_SEND_FENCE. Buffers are fixed registered offsets provided by caller.
extern "C" int probe_flags(void *h,const Meta *remote,uint64_t host_scratch,uint64_t host_seq,uint64_t remote_scratch,uint64_t remote_flags,uint64_t seq){return guard([&]{
 auto &e=*static_cast<Endpoint*>(h);if(host_scratch+e.qps.size()*8>e.bytes||host_seq+8>e.bytes)fail("local flag bounds");
 *reinterpret_cast<uint64_t*>(e.addr+host_seq)=seq;
 for(size_t q=0;q<e.qps.size();q++) {ibv_sge s{};s.addr=e.addr+host_scratch+q*8;s.length=8;s.lkey=e.mr->lkey;
 ibv_send_wr w{};w.sg_list=&s;w.num_sge=1;w.opcode=IBV_WR_RDMA_READ;w.send_flags=IBV_SEND_SIGNALED;w.wr.rdma.remote_addr=remote->addr+remote_scratch;w.wr.rdma.rkey=remote->rkey;post(e,q,w);}
 poll(e,e.qps.size());
 for(size_t q=0;q<e.qps.size();q++){ibv_sge s{};s.addr=e.addr+host_seq;s.length=8;s.lkey=e.mr->lkey;ibv_send_wr w{};w.sg_list=&s;w.num_sge=1;w.opcode=IBV_WR_RDMA_WRITE;w.send_flags=IBV_SEND_SIGNALED|IBV_SEND_FENCE|IBV_SEND_INLINE;w.wr.rdma.remote_addr=remote->addr+remote_flags+q*8;w.wr.rdma.rkey=remote->rkey;post(e,q,w);}poll(e,e.qps.size());
 });}
extern "C" void probe_close(void *h){auto *e=static_cast<Endpoint*>(h);if(!e)return;for(auto q:e->qps)ibv_destroy_qp(q);if(e->mr)ibv_dereg_mr(e->mr);if(e->cq)ibv_destroy_cq(e->cq);if(e->pd)ibv_dealloc_pd(e->pd);if(e->ctx)ibv_close_device(e->ctx);delete e;}
