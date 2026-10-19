"""Isolated prospective comparison. Public GET only. Does not access the original ETH ledger."""
import argparse, copy, fcntl, hashlib, json, math, os, sqlite3, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import core
ROOT=Path(__file__).resolve().parent
I=core.I;DAY=core.DAY
FILES=['runner.py','core.py','protocol.json','rules.json','frozen/paper.py','frozen/engine.py','frozen/protocol.json']

def enc(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def sha(x):return hashlib.sha256(x).hexdigest()
def now():return int(time.time()*1000)
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:f.write(value);f.flush();os.fsync(f.fileno())
    tmp.replace(path)
def hashes():return {f:sha((ROOT/f).read_bytes()) for f in FILES}
def check_files():
    expected=json.loads((ROOT/'build_manifest.json').read_text())['code_sha256']
    if hashes()!=expected:raise RuntimeError('Package source hash mismatch; do not reset or modify the running comparison')
    return expected

def get(pair,endpoint):
    if pair not in core.PAIRS or not (endpoint in ['depth','circuit_break_info'] or (endpoint.startswith('candlestick/15min/') and endpoint.split('/')[-1].isdigit())):
        raise ValueError('Only allowlisted public GET data endpoints are supported')
    url='https://public.bitbank.cc/'+pair+'/'+endpoint
    req=urllib.request.Request(url,headers={'User-Agent':'PaperComparison/1.0'},method='GET')
    with urllib.request.urlopen(req,timeout=15) as r:raw=r.read()
    obj=json.loads(raw)
    if obj.get('success')!=1:raise RuntimeError('Public data retrieval unsuccessful: '+pair+'/'+endpoint)
    return obj,raw

def validate_book(obj,circuit,observed):
    d=obj['data'];asks=[(float(p),float(q)) for p,q in d['asks']];bids=[(float(p),float(q)) for p,q in d['bids']]
    if not asks or not bids or not all(math.isfinite(x) and x>0 for row in asks+bids for x in row):raise ValueError('Invalid book')
    ask=min(p for p,q in asks);bid=max(p for p,q in bids);stamp=int(d['timestamp'])
    if not -5000<=observed-stamp<=90000 or not 0<bid<ask:raise ValueError('Stale or crossed book')
    if circuit['data']['mode']!='NONE':raise ValueError('Continuous trading unavailable')
    return {'observed_ms':observed,'timestamp':stamp,'mid':(ask+bid)/2,'ask':ask,'bid':bid,
            'ask_qty':sum(q for p,q in asks if p==ask),'bid_qty':sum(q for p,q in bids if p==bid),'spread_fraction':(ask-bid)/((ask+bid)/2)}

def collect(cache):
    start=now();slot=start//I*I;first=(slot//DAY-65)*DAY;today=slot//DAY*DAY;signals={};sources={}
    for pair in core.PAIRS:
        a=np.full(((slot-first)//I+1,6),np.nan);a[:,5]=first+np.arange(len(a))*I;fingerprints=[]
        for day in range(first,today+DAY,DAY):
            name=datetime.fromtimestamp(day/1000,timezone.utc).strftime('%Y%m%d');path=cache/pair/(name+'.json')
            if not path.exists() or day>=today-DAY:
                obj,raw=get(pair,'candlestick/15min/'+name);write(path,raw.decode());time.sleep(.11)
            else:raw=path.read_bytes();obj=json.loads(raw)
            if obj.get('success')!=1:raise ValueError('Invalid candle archive')
            fingerprints.append({'date':name,'sha256':sha(raw)})
            for group in obj['data']['candlestick']:
                if group['type']!='15min':raise ValueError('Unexpected interval')
                for row in group['ohlcv']:
                    t=int(row[5])
                    if first<=t<slot:
                        j=(t-first)//I
                        if t!=first+j*I:raise ValueError('Unaligned candle')
                        r=np.asarray(row,float)
                        if np.isfinite(a[j,0]) and not np.array_equal(a[j],r):raise ValueError('Conflicting candle')
                        a[j]=r
        b=a[:-1]
        if not np.all(np.isfinite(b)) or np.any(b[:,:4]<=0) or np.any(b[:,4]<0):raise ValueError('Incomplete 65-day candle window: '+pair)
        if np.any(b[:,2]>np.minimum(b[:,0],b[:,3])) or np.any(b[:,1]<np.maximum(b[:,0],b[:,3])):raise ValueError('Invalid OHLC')
        f,p=core.compute_signals(a);sig=core.signal_at(f,p,len(a)-1,slot)
        if not sig.get('valid'):raise ValueError('Incomplete indicators')
        sig['history_fingerprint']=sha(enc(fingerprints).encode());signals[pair]=sig;sources[pair]=fingerprints
    books={}
    for pair in core.PAIRS:
        circuit,_=get(pair,'circuit_break_info');depth,_=get(pair,'depth');books[pair]=validate_book(depth,circuit,now())
    observed=now()
    if observed//I*I!=slot:raise ValueError('Collection crossed interval boundary; no decisions made')
    for b in books.values():
        if not -5000<=observed-b['timestamp']<=90000:raise ValueError('Book expired during collection')
        b['observed_ms']=observed
    return {'slot':slot,'observed_ms':observed,'signals':signals,'books':books,'source_fingerprints':sources}

def initial(at,code_hashes):
    return {'version':'multiasset-comparison-v1','started_ms':at,'code_sha256':code_hashes,'last_slot':None,
            'health':'starting','last_error':None,'failures':0,'reported':[],
            'reference':core.eth.initial_state(at),'multi':{c:core.initial(at,2) for c in core.COSTS}}

def apply_tick(old,obs,rules):
    s=copy.deepcopy(old);events=[];notices=[];slot=obs['slot'];at=obs['observed_ms']
    if s['last_slot'] is not None and slot<=s['last_slot']:return s,[],[]
    if set(obs['signals'])!=set(core.PAIRS) or set(obs['books'])!=set(core.PAIRS):raise ValueError('Partial multi-asset input')
    for pair in core.PAIRS:
        if not obs['signals'][pair].get('valid') or obs['signals'][pair]['slot']!=slot:raise ValueError('Invalid or mixed-slot signal')
        b=obs['books'][pair]
        if b['observed_ms']!=at or not -5000<=at-b['timestamp']<=90000 or not 0<b['bid']<b['ask']:raise ValueError('Invalid synchronized quote')
    s['reference'],ev,_=core.eth.advance(s['reference'],obs['signals']['eth_jpy'],obs['books']['eth_jpy'])
    # The unchanged reference also calculates its original daily control accounts internally.
    events.extend(dict(e,scenario='reference') for e in ev if e['type']!='observation')
    for cost in core.COSTS:
        s['multi'][cost],ev=core.advance(s['multi'][cost],obs['signals'],obs['books'],rules,slot,at,cost)
        events.extend(dict(e,scenario='multi_'+cost) for e in ev)
    for e in events:
        if e['type']=='fill' or (e['type']=='paper_fill' and e['account'].startswith('candidate_')):
            notices.append({'kind':'paper_fill','scenario':e['scenario'],'account':e.get('account'),'pair':e.get('pair','eth_jpy'),'side':e['side'],'qty':e['qty'],'price':e['price']})
    oldgap=old['reference']['missing_slots'];gap=s['reference']['missing_slots']-oldgap
    if gap:notices.append({'kind':'monitoring_gap','missing_slots':gap,'backfilled':False})
    if old['health']=='error':notices.append({'kind':'data_recovered'})
    age=at-s['started_ms'];primary=s['multi']['base'];ref=s['reference']['accounts']['candidate_base']
    conditions={'90day_review':age>=90*DAY,'180day_review':age>=180*DAY and min([s['multi'][c]['round_trips'] for c in core.COSTS]+[s['reference']['accounts']['candidate_'+c]['round_trips'] for c in core.COSTS])>=20,
                'eth_drawdown_10pct':ref['max_drawdown']>=.1,'eth_drawdown_20pct':ref['max_drawdown']>=.2,'eth_doubled':ref['liquidation_value']>=20000,
                'multi_drawdown_10pct':primary['mdd']>=.1,'multi_drawdown_20pct':primary['mdd']>=.2,'multi_doubled':primary['liquidation']>=20000}
    for key,value in conditions.items():
        if value and key not in s['reported']:s['reported'].append(key);notices.append({'kind':key})
    s.update(last_slot=slot,health='ok',last_error=None)
    return s,events,notices

def error_state(old,message):
    s=copy.deepcopy(old);s.update(health='error',last_error=message,failures=old['failures']+1);return s

def connect(path,create=False):
    if not create and not path.exists():raise RuntimeError('Comparison not initialized; do not auto-reset')
    db=sqlite3.connect(path,timeout=30);db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS state(id INTEGER PRIMARY KEY CHECK(id=1),payload TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS journal(seq INTEGER PRIMARY KEY,payload TEXT NOT NULL,previous_hash TEXT NOT NULL,record_hash TEXT NOT NULL)')
    return db

def load(db):
    row=db.execute('SELECT payload FROM state WHERE id=1').fetchone()
    if not row:raise RuntimeError('Missing comparison state; do not reset')
    return json.loads(row[0])

def commit(db,s,record):
    record=dict(record,state_sha256=sha(enc(s).encode()));payload=enc(record)
    with db:
        row=db.execute('SELECT record_hash FROM journal ORDER BY seq DESC LIMIT 1').fetchone();prev=row[0] if row else ''
        db.execute('INSERT INTO journal(payload,previous_hash,record_hash) VALUES(?,?,?)',(payload,prev,sha((prev+'\n'+payload).encode())))
        db.execute('INSERT OR REPLACE INTO state(id,payload) VALUES(1,?)',(enc(s),))

def initialize(db,at,code_hashes):
    if db.execute('SELECT 1 FROM state').fetchone() or db.execute('SELECT 1 FROM journal').fetchone():raise RuntimeError('Already initialized; resetting is forbidden')
    s=initial(at,code_hashes);commit(db,s,{'type':'init','at':at,'code_sha256':code_hashes});return s

def verify(db,rules,code_hashes):
    previous='';s=None;records=0;fills=0
    ledgers={k:{'cash':10000.,'fees':0.,'qty':{p:0. for p in core.PAIRS}} for k in ['eth_base','eth_stress','multi_base','multi_stress']}
    for seq,payload,prev,digest in db.execute('SELECT seq,payload,previous_hash,record_hash FROM journal ORDER BY seq'):
        if prev!=previous or digest!=sha((prev+'\n'+payload).encode()):raise ValueError('Journal hash chain mismatch')
        r=json.loads(payload);ev=[]
        if r['type']=='init':
            if s is not None or r['code_sha256']!=code_hashes:raise ValueError('Invalid initialization or changed source')
            s=initial(r['at'],r['code_sha256'])
        elif r['type']=='tick':
            s,ev,_=apply_tick(s,r['observation'],rules)
            if ev!=r['events']:raise ValueError('Replay events differ')
        elif r['type']=='error':s=error_state(s,r['message'])
        else:raise ValueError('Unknown journal record')
        if r['state_sha256']!=sha(enc(s).encode()):raise ValueError('Replay state hash mismatch')
        for e in ev:
            if e['type']=='fill':key=e['scenario'];pair=e['pair'];due=e['due']
            elif e['type']=='paper_fill' and e['account'].startswith('candidate_'):key='eth_'+e['account'].split('_')[1];pair='eth_jpy';due=e['due_slot']
            else:continue
            if not e['decision_ms']<due<=e['at']:raise ValueError('Non-causal fill')
            l=ledgers[key];sign=1 if e['side']=='buy' else -1
            if abs(e['fee']-e['qty']*e['price']*.0015)>1e-8:raise ValueError('Fee mismatch')
            l['cash']-=sign*e['qty']*e['price']+e['fee'];l['qty'][pair]+=sign*e['qty'];l['fees']+=e['fee'];fills+=1
            if abs(l['cash']-e['cash_after'])>1e-6 or l['cash']<-.000001 or l['qty'][pair]<-1e-8:raise ValueError('Cash ledger mismatch')
        previous=digest;records+=1
    if s is None or enc(s)!=enc(load(db)):raise ValueError('Final state differs from replay')
    for key,l in ledgers.items():
        c=key.split('_')[1];a=s['reference']['accounts']['candidate_'+c] if key.startswith('eth_') else s['multi'][c]
        if abs(l['cash']-a['cash'])>1e-6 or abs(l['fees']-a['fees'])>1e-6:raise ValueError('Cash or fees do not reconcile')
        for p,q in l['qty'].items():
            actual=(a['qty'] if p=='eth_jpy' else 0.) if key.startswith('eth_') else a['positions'][p]['qty']
            if abs(q-actual)>1e-8:raise ValueError('Position does not reconcile')
    return {'ok':True,'records':records,'comparison_fills':fills,'last_hash':previous,'real_orders':False}

def summary(s):
    result={'started_jst':core.eth.jst(s['started_ms']),'last_slot_jst':core.eth.jst(s['last_slot']) if s['last_slot'] is not None else None,
            'last_observed_jst':core.eth.jst(s['reference']['last_observed_ms']) if s['reference']['last_observed_ms'] is not None else None,
            'health':s['health'],'last_error':s['last_error'],'missing_slots':s['reference']['missing_slots'],'failed_runs':s['failures'],'real_orders':False,'accounts':{}}
    for c in core.COSTS:
        a=s['reference']['accounts']['candidate_'+c];m=s['multi'][c]
        result['accounts']['eth_'+c]={'equity_yen':a['liquidation_value'],'pnl_yen':a['liquidation_value']-10000,'mdd':a['max_drawdown'],'round_trips':a['round_trips'],'orders':a['orders'],'fees_yen':a['fees']}
        result['accounts']['multi_'+c]={'equity_yen':m['liquidation'],'pnl_yen':m['liquidation']-10000,'mdd':m['mdd'],'round_trips':m['round_trips'],'orders':m['orders'],'fees_yen':m['fees'],'positions':{p:a['qty'] for p,a in m['positions'].items()},'minimum_lot_skipped_observations':m['lot_skips']}
    return result

def export(s,directory):
    data=summary(s);write(directory/'comparison_status.json',json.dumps(data,ensure_ascii=False,indent=2))
    rows=['複数銘柄とETHの仮想比較（実注文なし）','各方式を同時刻・仮想10,000円で開始。合算して投資する意味ではありません。',
          '開始 '+data['started_jst'],'最終確認 '+str(data['last_observed_jst']),f"状態 {data['health']} / 未確認区間 {data['missing_slots']} / 失敗回数 {data['failed_runs']}",
          '方式 | 売却費用控除後の資産 | 純損益 | 最大下落 | 往復数']
    for k,a in data['accounts'].items():rows.append(f"{k} | {a['equity_yen']:,.2f}円 | {a['pnl_yen']:+,.2f}円 | {a['mdd']:.2%} | {a['round_trips']}")
    rows+=['残高は仮定の売却費用込み。最大下落は観測時点のみ。税・サービス利用料を除く。','監視の空白を過去の注文で埋めない。元のETH運用とは別の比較台帳。90日固定・180日かつ両方式20往復が評価目安。']
    write(directory/'comparison_status.txt','\n'.join(rows)+'\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['preflight','init','tick','verify','status']);parser.add_argument('--state-dir',type=Path,default=ROOT/'state');args=parser.parse_args()
    code_hashes=check_files();rules=json.loads((ROOT/'rules.json').read_text());directory=args.state_dir.resolve();directory.mkdir(parents=True,exist_ok=True)
    with (directory/'comparison.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if args.command=='preflight':
            obs=collect(ROOT/'cache');write(directory/'preflight.json',enc({'at':now(),'code_sha256':code_hashes,'slot':obs['slot']}))
            print(enc({'ok':True,'readonly_strategy':True,'observed_jst':core.eth.jst(obs['observed_ms']),'pairs':list(obs['books']),'real_orders':False}));return
        path=directory/'comparison.sqlite3';db=connect(path,create=args.command=='init')
        if args.command=='init':
            pre=json.loads((directory/'preflight.json').read_text())
            if pre['code_sha256']!=code_hashes or not 0<=now()-pre['at']<I:raise RuntimeError('Run a successful recent preflight before initialization')
            s=initialize(db,now(),code_hashes);export(s,directory);print(enc({'initialized':True,'summary':summary(s)}));return
        if args.command=='verify':print(enc(verify(db,rules,code_hashes)));return
        old=load(db)
        if old['code_sha256']!=code_hashes:raise ValueError('Running comparison source changed')
        if args.command=='status':print(enc(summary(old)));return
        slot=now()//I*I
        if old['last_slot'] is not None and slot<=old['last_slot']:print(enc({'duplicate_slot':True,'notices':[]}));return
        # Catch collection errors only. Failed accounting, commits or exports must not overwrite a committed state.
        try:obs=collect(ROOT/'cache')
        except Exception as exc:
            message=type(exc).__name__+': '+str(exc);s=error_state(old,message);commit(db,s,{'type':'error','at':now(),'message':message});export(s,directory)
            print(enc({'ok':False,'notices':[{'kind':'data_error','message':message}] if old['health']!='error' else [],'repeated_error':old['health']=='error'}));return
        s,events,notices=apply_tick(old,obs,rules);commit(db,s,{'type':'tick','observation':obs,'events':events})
        if any(x['kind']=='paper_fill' for x in notices):verify(db,rules,code_hashes)
        export(s,directory);print(enc({'ok':True,'notices':notices,'summary':summary(s)}))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(enc({'ok':False,'fatal':True,'message':type(exc).__name__+': '+str(exc),'instruction':'Preserve comparison ledger; report failure; never reinitialize'}));raise SystemExit(1)
