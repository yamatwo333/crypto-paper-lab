"""Causal multi-asset paper ledger. One cash balance per alternative. No networking."""
import copy, importlib.util, math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('frozen_eth_paper',ROOT/'frozen/paper.py')
eth=importlib.util.module_from_spec(spec);spec.loader.exec_module(eth)
spec=importlib.util.spec_from_file_location('frozen_signal_engine',ROOT/'frozen/engine.py')
signal_engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(signal_engine)
PAIRS=['btc_jpy','eth_jpy','xrp_jpy','sol_jpy'];I=900000;DAY=86400000
COSTS={'base':(.0015,.0005),'stress':(.0015,.0035)}

def rounded_price(p,tick,buy):
    return (math.ceil(p/tick) if buy else math.floor(p/tick))*tick

def units(value,step):
    return max(0.,math.floor(value/step+1e-10)*step)

def position():
    return {'qty':0.,'basis':0.,'entry_ms':None,'entry_price':None,'highwater':None,
            'stop_distance':None,'trail_distance':None,'last_exit_slot':None,
            'confirmation':0,'pending_exit':None,'last_mid':None,'last_bid':None}

def initial(start,max_positions=2):
    return {'started_ms':start,'last_slot':None,'cash':10000.,'fees':0.,'orders':0,
            'round_trips':0,'realized_pnl':0.,'winning_trips':0,'max_positions':max_positions,
            'positions':{p:position() for p in PAIRS},'pending_buys':[],
            'peak':10000.,'mdd':0.,'equity':10000.,'liquidation':10000.,
            'exposure':0.,'missing_slots':0,'lot_skips':0,'budget_skips':0}

def prices(book,rule,slip):
    tick=10**(-rule['price_digits'])
    return (rounded_price(book['ask']*(1+slip),tick,True),
            rounded_price(book['bid']*(1-slip),tick,False))

def mark(s,books,rules,fee,slip):
    market=0.;liquid=0.
    for pair,a in s['positions'].items():
        b=books.get(pair)
        if b:
            a['last_mid']=b['mid'];a['last_bid']=b['bid']
        if a['qty']:
            if a['last_mid'] is None:raise RuntimeError('No valuation price for held asset')
            market+=a['qty']*a['last_mid']
            sell=rounded_price(a['last_bid']*(1-slip),10**(-rules[pair]['price_digits']),False)
            liquid+=a['qty']*sell*(1-fee)
    s['equity']=s['cash']+market;s['liquidation']=s['cash']+liquid
    s['exposure']=market/s['equity'] if s['equity'] else 0.
    return market

def advance(state,signals,books,rules,slot,observed,cost='base'):
    s=copy.deepcopy(state);fee,slip=COSTS[cost];events=[]
    if observed//I*I!=slot:raise ValueError('Observation crossed a slot boundary')
    if s['last_slot'] is not None and slot<=s['last_slot']:return s,[]
    last=s['last_slot'] if s['last_slot'] is not None else s['started_ms']//I*I-I
    gap=max(0,(slot-last)//I-1)
    if gap:
        s['missing_slots']+=gap;s['pending_buys']=[]
        for a in s['positions'].values():a['confirmation']=0
        events.append({'type':'gap','slots':gap,'at':observed})
    mark(s,books,rules,fee,slip)
    filled=set()
    # Existing exits consume no new capital; process them before prior buy intents.
    for pair,a in s['positions'].items():
        intent=a['pending_exit'];b=books.get(pair)
        if not intent or slot<intent['due'] or not b:continue
        if a['qty']>b['bid_qty']+1e-12:continue
        _,price=prices(b,rules[pair],slip);qty=a['qty'];charge=qty*price*fee
        pnl=qty*price-charge-a['basis'];s['cash']+=qty*price-charge
        s['fees']+=charge;s['orders']+=1;s['round_trips']+=1;s['realized_pnl']+=pnl
        s['winning_trips']+=int(pnl>0)
        events.append({'type':'fill','at':observed,'pair':pair,'side':'sell','qty':qty,'price':price,
                       'fee':charge,'pnl':pnl,'reason':intent['reason'],'decision_ms':intent['decision_ms'],
                       'due':intent['due'],'cash_after':s['cash'],'real_order':False})
        old_mid,old_bid=a['last_mid'],a['last_bid'];a.update(position())
        a.update(last_exit_slot=slot,last_mid=old_mid,last_bid=old_bid)
    remaining=[]
    for intent in s['pending_buys']:
        pair=intent['pair'];a=s['positions'][pair];b=books.get(pair)
        if slot<intent['due']:remaining.append(intent);continue
        if observed>intent['due']+I:continue
        if not b or b['spread_fraction']>.005:remaining.append(intent);continue
        if a['qty']:continue
        occupied=sum(p['qty']>0 for p in s['positions'].values())
        if occupied>=s['max_positions']:continue
        market=mark(s,books,rules,fee,slip);eq=s['equity']
        per=.25/s['max_positions'];buy,_=prices(b,rules[pair],slip)
        # The 25% constraint is imposed on actual pre-cost market exposure after fees,
        # while per-entry notional including fees is limited to the fixed sleeve size.
        gap=max(0.,.25*eq-market)
        cap_units=gap/(b['mid']+.25*(buy*(1+fee)-b['mid']))
        quantity=units(min(per*eq/(buy*(1+fee)),s['cash']/(buy*(1+fee)),cap_units),rules[pair]['unit_amount'])
        if quantity<rules[pair]['unit_amount']-1e-12:
            s['lot_skips']+=1;continue
        if quantity>b['ask_qty']+1e-12:remaining.append(intent);continue
        charge=quantity*buy*fee;outlay=quantity*buy+charge
        s['cash']-=outlay;s['fees']+=charge;s['orders']+=1
        a.update(qty=quantity,basis=outlay,entry_ms=observed,entry_price=buy,highwater=buy,
                 stop_distance=max(.02*buy,3*intent['atr']),trail_distance=max(.02*buy,4*intent['atr']),confirmation=0)
        filled.add(pair)
        events.append({'type':'fill','at':observed,'pair':pair,'side':'buy','qty':quantity,'price':buy,
                       'fee':charge,'reason':'ranked_entry','decision_ms':intent['decision_ms'],'due':intent['due'],
                       'cash_after':s['cash'],'pre_entry_equity':eq,'outlay_fraction':outlay/eq,'real_order':False})
    s['pending_buys']=remaining
    candidates=[];pending_pairs={p['pair'] for p in remaining}
    for pair,a in s['positions'].items():
        f=signals.get(pair)
        if not f or not f.get('valid',True):a['confirmation']=0;continue
        if a['qty']:
            a['confirmation']=0
            if a['pending_exit'] or pair in filled:continue
            a['highwater']=max(a['highwater'],f['p']);reason=None
            if f['p']<=a['entry_price']-a['stop_distance']:reason='initial_stop'
            elif f['p']<=a['highwater']-a['trail_distance']:reason='trailing_exit'
            elif f['exit']:reason='momentum_exit'
            elif observed-a['entry_ms']>=60*DAY:reason='time_exit'
            if reason:a['pending_exit']={'due':slot+I,'decision_ms':observed,'reason':reason}
        else:
            if pair in pending_pairs:continue
            cooldown=a['last_exit_slot'] is not None and slot-a['last_exit_slot']<DAY
            if f['entry'] and f['gate'] and not cooldown:
                a['confirmation']=min(4,a['confirmation']+1)
                if a['confirmation']>=4:
                    score=f['mom30']/max(f['d_atr']/f['p'],1e-12)
                    candidates.append((score,pair,f['h4_atr']))
            else:a['confirmation']=0
    free=s['max_positions']-sum(a['qty']>0 for a in s['positions'].values())-len(remaining)
    # Skip currently unaffordable minimum lots before choosing; recheck at execution.
    market=mark(s,books,rules,fee,slip);eq=s['equity']
    for score,pair,atr in sorted(candidates,key=lambda x:(-x[0],x[1])):
        if free<=0:break
        b=books.get(pair)
        if not b:continue
        buy,_=prices(b,rules[pair],slip)
        if rules[pair]['unit_amount']*buy*(1+fee)>(.25/s['max_positions'])*eq+1e-9:
            s['lot_skips']+=1;continue
        if .25*eq-market<=0:s['budget_skips']+=1;continue
        s['pending_buys'].append({'pair':pair,'due':slot+I,'decision_ms':observed,'atr':atr,'score':score})
        s['positions'][pair]['confirmation']=0;free-=1
    mark(s,books,rules,fee,slip)
    s['peak']=max(s['peak'],s['liquidation']);s['mdd']=max(s['mdd'],1-s['liquidation']/s['peak'])
    s['last_slot']=slot
    assert s['cash']>=-1e-7
    assert sum(a['qty']>0 for a in s['positions'].values())<=s['max_positions']
    return s,events

def compute_signals(candles):
    f,_=signal_engine.features(candles,with_models=False)
    p=signal_engine.make_plan(f,'buffer_momentum_S')
    return f,p

def signal_at(f,p,i,slot):
    if not f['valid'][i]:return {'slot':slot,'valid':False}
    v={k:float(f[k][i]) for k in ['p','mom30','d_atr','h4_atr','h1_atr','d_ma5','d_ma20','h4_ma5','h4_ma20']}
    v.update(slot=slot,valid=True,entry=bool(p['entry'][i]),exit=bool(p['exit'][i]),gate=bool(p['gate'][i]),
             daily_on=bool(f['daily_on'][i]),boundary_band=max(.01,.5*v['d_atr']/v['p']))
    return v
