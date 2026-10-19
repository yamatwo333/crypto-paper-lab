"""Causal 15-minute observations with independent entry, exit and wait decisions.
Research only. No order endpoint, credentials, scheduler or paid model API.
"""
import hashlib,json,math
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;P=json.loads((ROOT/'protocol.json').read_text())
MS=900000;DAY=86400000
FAMILIES=['buffer_momentum','trend_agreement','breakout','pullback','range_reversion','daily_fast_guard','consensus']
NAMES=[f'{f}_{s}' for f in FAMILIES for s in ['L','S']]+['ridge_10','ridge_100']
assert len(NAMES)==P['candidate_count']==16

def ts(s):return int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()*1000)
def iso(t):return datetime.fromtimestamp(t/1000,timezone.utc).isoformat()
def save(name,obj): (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
def shift(x,k=1):return pd.Series(x).shift(k).to_numpy()
def mean(x,k):return pd.Series(x).rolling(k).mean().to_numpy()
def std(x,k):return pd.Series(x).rolling(k).std(ddof=0).to_numpy()

def expand(x,factor,n):
    # Boundary i sees only aggregate bars ending at or before i. Do not ffill across invalid bars.
    ids=np.arange(n)//factor-1;out=np.full(n,np.nan);good=ids>=0;out[good]=x[ids[good]];return out

def aggregate(a,factor):
    b=a[:len(a)//factor*factor].reshape(-1,factor,6)
    valid=np.all(np.isfinite(b[:,:,0]),axis=1);active=b[:,:,4]>0
    # Ignore carried OHLC from zero-volume bars when there are actual trades in the group.
    out=np.full((len(b),6),np.nan)
    for j,rows in enumerate(b):
        if not valid[j]:continue
        traded=rows[active[j]]
        if len(traded):out[j,:5]=[traded[0,0],traded[:,1].max(),traded[:,2].min(),traded[-1,3],rows[:,4].sum()]
        else:out[j,:5]=[rows[-1,3]]*4+[0.]
        out[j,5]=rows[0,5]
    return out

def features(a,with_models=True):
    n=len(a);o,h,l,c,v,t=a.T;p=shift(c);f={'p':p,'prevp':shift(p),'prev2p':shift(p,2),'v':shift(v)}
    f['mom1']=np.log(p/shift(p,96));f['mom7']=np.log(p/shift(p,672));f['mom30']=p/shift(p,2880)-1
    f['volume24']=shift(mean(v,96),2)
    for factor,prefix in [(4,'h1'),(16,'h4'),(96,'d')]:
        b=aggregate(a,factor);bo,bh,bl,bc,bv,bt=b.T
        prev=shift(bc);tr=np.maximum(bh-bl,np.maximum(abs(bh-prev),abs(bl-prev)))
        fields={'close':bc,'atr':mean(tr,14),'ma5':mean(bc,5),'ma20':mean(bc,20)}
        if prefix=='d':fields.update({'ma60':mean(bc,60),'ma20prev5':shift(mean(bc,20),5),'vol20':std(np.log(bc/prev),20)})
        if prefix=='h1':fields.update({'ma48':mean(bc,48),'std48':std(bc,48)})
        for key,x in fields.items():f[prefix+'_'+key]=expand(x,factor,n)
    highs=np.where(v>0,h,np.nan);lows=np.where(v>0,l,np.nan)
    f['high72']=pd.Series(highs).rolling(288,min_periods=1).max().shift(2).to_numpy()
    f['low36']=pd.Series(lows).rolling(144,min_periods=1).min().shift(2).to_numpy()
    f['valid']=(pd.Series(np.isfinite(c).astype(float)).rolling(5760).sum().shift(1).to_numpy()==5760)
    f['valid']&=np.isfinite(f['d_ma60'])&np.isfinite(f['h4_atr'])&np.isfinite(f['h1_atr'])&np.isfinite(p)
    f['daily_on']=f['d_ma5']>f['d_ma20'];f['h4_on']=f['h4_ma5']>f['h4_ma20']
    model_audit={}
    if with_models:
        X=np.column_stack([f['mom1'],f['mom7'],np.log1p(f['mom30']),
                           f['d_ma5']/f['d_ma20']-1,f['d_vol20'],p/f['h1_ma20']-1,
                           f['h4_ma5']/f['h4_ma20']-1])
        for penalty in (10,100):
            f['ridge_'+str(penalty)],model_audit[str(penalty)]=ridge_predictions(X,p,t,penalty)
    return f,model_audit

def ridge_predictions(X,p,t,penalty):
    n=len(p);out=np.full(n,np.nan);audits=[];beta=None;mu=None;sd=None;intercept=0.
    days=np.arange(0,n,96);fit_points=[]
    for i in days:
        d=datetime.fromtimestamp(t[i]/1000,timezone.utc)
        if d.day==1 and d.month in (1,4,7,10):fit_points.append(i)
    for j,i in enumerate(fit_points):
        end=fit_points[j+1] if j+1<len(fit_points) else n
        train=days[(days>=i-730*96)&(days+96<=i)]
        train=train[np.all(np.isfinite(X[train]),axis=1)&np.isfinite(p[train])&np.isfinite(p[train+96])]
        if len(train)<180:continue
        xx=X[train];y=np.log(p[train+96]/p[train]);mu=xx.mean(axis=0);sd=xx.std(axis=0);sd=np.maximum(sd,1e-8)
        z=(xx-mu)/sd;intercept=float(y.mean());beta=np.linalg.solve(z.T@z+penalty*np.eye(X.shape[1]),z.T@(y-intercept))
        good=np.all(np.isfinite(X[i:end]),axis=1);pred=(X[i:end][good]-mu)/sd@beta+intercept
        chunk=out[i:end];chunk[good]=np.clip(pred,-.2,.2)
        audits.append({'fit_utc':iso(t[i]),'training_first_utc':iso(t[train[0]]),'training_last_feature_utc':iso(t[train[-1]]),
                       'last_label_available_utc':iso(t[train[-1]+96]),'samples':len(train),'penalty':penalty,
                       'feature_mean':mu.tolist(),'feature_sd':sd.tolist(),'coef':beta.tolist(),'intercept':intercept})
        assert train[-1]+96<=i
    return out,audits

class Market:
    def __init__(self,pair='btc_jpy',end='2026-10-18'):
        self.pair=pair;self.first=ts(P['data'][pair+'_from']);last=ts(end);n=(last-self.first)//MS
        a=np.full((n,6),np.nan);a[:,5]=np.arange(self.first,last,MS);duplicates=0;files=0
        for path in sorted((ROOT/'data'/pair).glob('????????.json')):
            if path.stem>=end.replace('-',''):continue
            files+=1;obj=json.loads(path.read_text());assert obj['success']==1
            for candle in obj['data']['candlestick']:
                for row in candle['ohlcv']:
                    r=np.array(row,float);idx=int((r[5]-self.first)//MS)
                    if not 0<=idx<n:continue
                    assert r[5]==a[idx,5]
                    if np.isfinite(a[idx,0]):duplicates+=1;assert np.array_equal(a[idx],r)
                    a[idx]=r
        valid=np.isfinite(a[:,0]);b=a[valid]
        assert np.all(np.isfinite(b)) and np.all(b[:,:4]>0) and np.all(b[:,4]>=0)
        assert np.all(b[:,2]<=np.minimum(b[:,0],b[:,3])) and np.all(b[:,1]>=np.maximum(b[:,0],b[:,3]))
        self.a=a;self.o,self.h,self.l,self.c,self.v,self.t=a.T;self.valid=valid
        self.f,self.model_audit=features(a);self.plans={}
        self.audit={'pair':pair,'files':files,'first_utc':iso(self.first),'end_exclusive_utc':end,
                    'bars':int(valid.sum()),'expected_bars':n,'missing_bars':int((~valid).sum()),
                    'missing_utc':[iso(t) for t in self.t[~valid]],'zero_volume_bars':int(np.sum(self.v==0)),
                    'identical_duplicates':duplicates,'aggregate_zero_volume_policy':'Use traded-only OHLC within complete groups; carry the last recorded close for all-zero groups.'}
    def bounds(self,start,end):return (ts(start)-self.first)//MS,(ts(end)-self.first)//MS
    def plan(self,name,threshold_scale=1.):
        key=(name,threshold_scale)
        if key not in self.plans:self.plans[key]=make_plan(self.f,name,threshold_scale)
        return self.plans[key]

def make_plan(f,name,scale=1.):
    if name.startswith('ridge_'):family='ridge';strict=True
    else:family,setting=name.rsplit('_',1);strict=setting=='S'
    persistence=4 if strict else 2;cooldown=96 if strict else 24;stop=3 if strict else 2;trail=4 if strict else 3;mult=4 if strict else 3
    p=f['p'];band=np.maximum(.01 if strict else .005,(.5 if strict else .25)*f['d_atr']/p)*scale
    daily=f['daily_on'];regime=f['d_close']>f['d_ma60'];gap=f['h4_ma5']/f['h4_ma20']-1
    if family=='buffer_momentum':entry=(f['mom30']>band)&f['h4_on'];leave=f['mom30']<-band/2
    elif family=='trend_agreement':entry=daily&regime&(gap>band/2);leave=(f['d_ma5']<f['d_ma20']*(1-band/2))|(f['h4_close']<f['h4_ma20']*(1-band))
    elif family=='breakout':
        entry=(p>f['high72']+(.25 if strict else .1)*scale*f['h4_atr'])&regime&(f['v']>=(1.5 if strict else 1.)*f['volume24'])
        leave=p<f['low36'];persistence=1
    elif family=='pullback':
        dip=p<f['h1_ma20']-(1. if strict else .5)*scale*f['h1_atr']
        armed=pd.Series(dip.astype(float)).rolling(96,min_periods=96).max().to_numpy()>0
        entry=daily&regime&armed&(p>f['h1_ma20']);leave=~daily
    elif family=='range_reversion':
        sideways=(abs(f['d_ma20']/f['d_ma20prev5']-1)<.02)&(abs(f['h4_ma5']-f['h4_ma20'])<.5*f['h4_atr'])
        z=(p-f['h1_ma48'])/np.maximum(f['h1_std48'],1e-12)
        entry=sideways&(z<-(2. if strict else 1.5)*scale)&(p>f['prevp'])&(f['prevp']>f['prev2p'])&((f['h1_ma48']/p-1)>=mult*.004)
        leave=p>=f['h1_ma48'];persistence=1
    elif family=='daily_fast_guard':entry=daily&(p>f['h4_ma20']);leave=~daily
    elif family=='consensus':
        votes=(f['mom30']>band).astype(int)+(daily&regime).astype(int)+(gap>band/2).astype(int)
        entry=votes>=(3 if strict else 2);leave=votes<=1
    elif family=='ridge':entry=f[name]>.012;leave=f[name]<0
    else:raise ValueError(name)
    opportunity=2*f['h4_atr']/p>=mult*.004
    no_shock=(abs(p-f['prevp'])<=2*f['h1_atr'])&(f['h4_atr']/p<=.08)&(f['v']>0)
    return {'entry':entry&f['valid'],'exit':leave&f['valid'],'gate':opportunity&no_shock&f['valid'],
            'valid':f['valid'],'confirmation':persistence,'cooldown':cooldown,'stop_atr':stop,'trail_atr':trail,
            'max_hold':48*4 if family=='range_reversion' else 60*96,'family':family,'name':name}

def simulate(m,start,end,name,cost='base',step=.0001,allocation=.25,initial=10000.,lag=1,details=False,ablation=None,threshold_scale=1.):
    if cost=='current_fee_proxy':fee=P['costs'][cost]['btc_fee' if m.pair=='btc_jpy' else 'eth_fee'];slip=P['costs'][cost]['slippage']
    else:fee=P['costs'][cost]['fee'];slip=P['costs'][cost]['slippage']
    benchmark=name in ['cash','hold25','hold100','unbuffered_momentum']
    plan=None if benchmark else m.plan(name,threshold_scale)
    if name=='hold100':allocation=1.
    p=m.f['p'];atr=m.f['h4_atr'];n=end-start
    cash=float(initial);units=0.;pending=None;last_exit=-10**9;entry_i=None;entry_price=None;entry_cost=0.;highwater=0.;stopdist=0.;traildist=0.;confirm=0
    equity=np.empty(n);exposure=np.empty(n);orders=[];trades=[];fees=0.;peak=initial;maxdd=0.;lowdd=0.;hold_obs=0;wait_obs=0;blocked_obs=0;entry_orders=0;exit_orders=0;lastmark=m.c[start-1]
    cooldown=0 if plan is None or ablation=='no_cooldown' else plan['cooldown']
    persistence=1 if plan is None or ablation=='no_confirmation' else plan['confirmation']
    expected_stop=2 if plan is None else plan['stop_atr'];expected_trail=3 if plan is None else plan['trail_atr']
    healthy=m.f['valid'];entry_arr=None if plan is None else plan['entry'];exit_arr=None if plan is None else plan['exit'];gate_arr=None if plan is None else plan['gate']

    def fill(i,side,reason,known_atr,price=None,decision=None,final=False):
        nonlocal cash,units,fees,entry_i,entry_price,entry_cost,highwater,stopdist,traildist,last_exit,entry_orders,exit_orders,confirm
        ref=float(m.o[i] if price is None else price)
        value=math.ceil(ref*(1+slip)) if side=='buy' else math.floor(ref*(1-slip))
        if side=='buy':
            quantity=allocation*cash/(value*(1+fee))
            if step:quantity=math.floor((quantity+1e-12)/step)*step
            if quantity<1e-10:return False
            f=quantity*value*fee;cash-=quantity*value+f;units=quantity;fees+=f
            entry_i=i;entry_price=value;entry_cost=quantity*value+f;highwater=value
            stopdist=max(.02*value,expected_stop*known_atr);traildist=max(.02*value,expected_trail*known_atr);entry_orders+=1
        else:
            quantity=units
            if quantity<1e-10:return False
            f=quantity*value*fee;cash+=quantity*value-f;fees+=f
            pnl=quantity*value-f-entry_cost
            trades.append({'entry_bar':entry_i,'exit_bar':i,'entry_utc':iso(m.t[entry_i]),'exit_utc':iso(m.t[i]+(MS if final else 0)),
                           'qty':quantity,'net_pnl_yen':pnl,'net_return_on_deployed_cash':pnl/entry_cost,'holding_hours':(i-entry_i)/4,'exit_reason':reason})
            units=0.;entry_i=None;last_exit=i;exit_orders+=1
        orders.append({'bar':int(i),'utc':iso(m.t[i]+(MS if final else 0)),'decision_bar':decision,'side':side,'qty':quantity,'price':value,'fee':f,'reason':reason,'final':final})
        confirm=0;assert cash>=-1e-6 and units>=0
        return True

    for i in range(start,end):
        tradable=bool(m.valid[i] and m.v[i]>0);new_entry=False
        if pending is not None and i>=pending['due'] and tradable:
            new_entry=fill(i,pending['side'],pending['reason'],pending['atr'],decision=pending['decision']) and pending['side']=='buy'
            pending=None
        if pending is None and not new_entry and healthy[i]:
            if units>0:
                highwater=max(highwater,p[i]);reason=None
                if plan is None:
                    if name=='unbuffered_momentum' and m.f['mom30'][i]<=0:reason='momentum_off'
                else:
                    if ablation!='no_protection' and p[i]<=entry_price-stopdist:reason='initial_stop'
                    elif ablation!='no_protection' and p[i]<=highwater-traildist:reason='trailing_exit'
                    elif exit_arr[i]:reason='rule_exit'
                    elif i-entry_i>=plan['max_hold']:reason='time_exit'
                if reason:pending={'side':'sell','reason':reason,'atr':0.,'due':i+lag,'decision':i}
                else:hold_obs+=1
            else:
                raw=(name in ['hold25','hold100'] and entry_orders==0) or (name=='unbuffered_momentum' and m.f['mom30'][i]>0) if plan is None else bool(entry_arr[i])
                gates=True if plan is None or ablation=='no_gates' else bool(gate_arr[i])
                if raw and not gates:blocked_obs+=1
                if raw and gates and i-last_exit>=cooldown:
                    confirm+=1
                    if confirm>=persistence:pending={'side':'buy','reason':'entry_conditions','atr':float(atr[i]),'due':i+lag,'decision':i}
                    else:wait_obs+=1
                else:confirm=0;wait_obs+=1
        elif pending is None and not new_entry:
            confirm=0;wait_obs+=1
        if pending is not None and pending['due']==i and tradable:
            fill(i,pending['side'],pending['reason'],pending['atr'],decision=pending['decision']);pending=None
        if m.valid[i]:lastmark=float(m.c[i])
        val=cash+units*lastmark;exposure[i-start]=units*lastmark/val if val else 0.
        if m.valid[i]:lowdd=max(lowdd,1-(cash+units*m.l[i])/peak)
        if i==end-1 and units>0:
            assert tradable,'Final bar not tradable'
            fill(i,'sell','final_liquidation',0.,price=lastmark,decision=None,final=True);val=cash
        peak=max(peak,val);maxdd=max(maxdd,1-val/peak);equity[i-start]=val
    ids=np.unique(np.r_[np.arange(95,n,96),n-1]).astype(int);daily=equity[ids]
    years=np.array([datetime.fromtimestamp(m.t[start+k]/1000,timezone.utc).year for k in ids]);annual={};prev=initial
    for year in sorted(set(years)):
        val=daily[np.where(years==year)[0][-1]];annual[str(year)]=float(val/prev-1);prev=val
    gains=[t['net_pnl_yen'] for t in trades if t['net_pnl_yen']>0];losses=[t['net_pnl_yen'] for t in trades if t['net_pnl_yen']<0]
    result={'name':name,'pair':m.pair,'start':iso(m.t[start]),'end_exclusive':iso(m.t[end-1]+MS),'days':n/96,
            'cost':cost,'allocation':allocation,'lot_step':step,'initial_yen':initial,'lag_minutes':lag*15,
            'final_yen':float(equity[-1]),'total_return':float(equity[-1]/initial-1),
            'cagr':float((equity[-1]/initial)**(365.25/(n/96))-1),'mdd':float(maxdd),'low_mark_drawdown':float(lowdd),
            'orders':len(orders),'round_trips':len(trades),'win_rate':len(gains)/len(trades) if trades else None,
            'profit_factor':sum(gains)/-sum(losses) if losses else None,'mean_trade_pnl_yen':float(np.mean([t['net_pnl_yen'] for t in trades])) if trades else None,
            'median_holding_hours':float(np.median([t['holding_hours'] for t in trades])) if trades else None,
            'fees_yen':float(fees),'mean_exposure':float(exposure.mean()),'annual_returns':annual,
            'observations':n,'wait_observations':wait_obs,'hold_observations':hold_obs,'entry_gate_blocked_observations':blocked_obs,
            'no_order_fraction':1-len(orders)/n,'ablation':ablation,'threshold_scale':threshold_scale}
    result['score']=result['cagr']-.5*maxdd
    if details:result.update({'daily_dates':[iso(m.t[start+k]+MS) for k in ids],'daily_equity':daily.tolist(),'orders_detail':orders,'trades':trades,'_full_equity':equity})
    return result

def clean(r):return {k:v for k,v in r.items() if not k.startswith('_')}
