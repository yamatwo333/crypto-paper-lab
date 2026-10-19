import copy, hashlib, json, sqlite3, tempfile, unittest
from pathlib import Path
import numpy as np
import core, runner
I=core.I;T=100000*I
RULES={p:{'unit_amount':.0001,'price_digits':1 if p=='sol_jpy' else 3 if p=='xrp_jpy' else 0} for p in core.PAIRS}
def signal(slot,entry=True,exit=False,price=1000.,score=.2):
    return {'valid':True,'slot':slot,'p':price,'mom30':score,'d_atr':20.,'h4_atr':20.,'h1_atr':10.,'d_ma5':price,'d_ma20':price,'h4_ma5':price,'h4_ma20':price,'entry':entry,'exit':exit,'gate':True,'daily_on':False,'boundary_band':.01}
def book(slot,price=1000.):return {'observed_ms':slot+1000,'timestamp':slot,'mid':price,'ask':price+1,'bid':price-1,'ask_qty':1e6,'bid_qty':1e6,'spread_fraction':2/price}
def observation(slot,entry=True,exit=False,price=1000.):
    return {'slot':slot,'observed_ms':slot+1000,'signals':{p:signal(slot,entry,exit,price) for p in core.PAIRS},'books':{p:book(slot,price) for p in core.PAIRS}}
class ComparisonTests(unittest.TestCase):
    def test_unchanged_reference(self):
        self.assertEqual(hashlib.sha256((core.ROOT/'frozen/paper.py').read_bytes()).hexdigest(),'5da5a2799eec560aef2bed8a964afa1095ca52e4b13df02cc7567b6690d20d42')
        a=core.eth.initial_state(T);s=runner.initial(T,{})
        for k in range(9):
            o=observation(T+k*I,exit=k>=6)
            a,_,_=core.eth.advance(a,o['signals']['eth_jpy'],o['books']['eth_jpy']);s,_,_=runner.apply_tick(s,o,RULES)
            self.assertEqual(a,s['reference'])
    def test_single_asset_core_matches_eth_v1(self):
        for cost in core.COSTS:
            s=core.initial(T,1);r=core.eth.initial_state(T)
            for k in list(range(12))+list(range(15,120)):
                price=1000+100*np.sin(k/12)
                o=observation(T+k*I,exit=k in [7,110],price=price)
                signals={'eth_jpy':o['signals']['eth_jpy']};books={'eth_jpy':o['books']['eth_jpy']}
                s,_=core.advance(s,signals,books,RULES,o['slot'],o['observed_ms'],cost)
                r,_,_=core.eth.advance(r,signals['eth_jpy'],books['eth_jpy'])
                a=r['accounts']['candidate_'+cost]
                for key in ['cash','fees','orders','round_trips']:
                    self.assertAlmostEqual(s[key],a[key],places=7)
                self.assertAlmostEqual(s['positions']['eth_jpy']['qty'],a['qty'],places=9)
                self.assertAlmostEqual(s['liquidation'],a['liquidation_value'],places=7)
    def test_four_confirmations_next_slot_and_budget(self):
        s=core.initial(T)
        for k in range(4):
            o=observation(T+k*I);s,e=core.advance(s,o['signals'],o['books'],RULES,o['slot'],o['observed_ms'])
            self.assertFalse(any(x['type']=='fill' for x in e))
        self.assertEqual([x['pair'] for x in s['pending_buys']],['btc_jpy','eth_jpy'])
        o=observation(T+4*I,price=1100);s,e=core.advance(s,o['signals'],o['books'],RULES,o['slot'],o['observed_ms'])
        fills=[x for x in e if x['type']=='fill'];self.assertEqual(len(fills),2)
        self.assertTrue(all(x['price']>1100 and x['outlay_fraction']<=.125+1e-10 for x in fills));self.assertLessEqual(s['exposure'],.25+1e-10)
        self.assertGreaterEqual(s['cash'],7500)
    def test_btc_minimum_lot_is_skipped(self):
        s=core.initial(T)
        for k in range(4):
            o=observation(T+k*I);o['books']['btc_jpy']=book(o['slot'],13069926);o['signals']['btc_jpy']['mom30']=10
            s,_=core.advance(s,o['signals'],o['books'],RULES,o['slot'],o['observed_ms'])
        self.assertNotIn('btc_jpy',[x['pair'] for x in s['pending_buys']]);self.assertGreater(s['lot_skips'],0)
    def test_gap_cancels_entry_but_retains_exit(self):
        s=core.initial(T)
        for k in range(4):
            o=observation(T+k*I);s,_=core.advance(s,o['signals'],o['books'],RULES,o['slot'],o['observed_ms'])
        o=observation(T+6*I);s,e=core.advance(s,o['signals'],o['books'],RULES,o['slot'],o['observed_ms']);self.assertEqual(s['orders'],0);self.assertEqual(s['missing_slots'],2)
        for k in range(7,11):
            o=observation(T+k*I);s,_=core.advance(s,o['signals'],o['books'],RULES,o['slot'],o['observed_ms'])
        o=observation(T+11*I,entry=False,exit=True);s,_=core.advance(s,o['signals'],o['books'],RULES,o['slot'],o['observed_ms'])
        o=observation(T+15*I,entry=False,exit=True,price=900);s,e=core.advance(s,o['signals'],o['books'],RULES,o['slot'],o['observed_ms'])
        sells=[x for x in e if x['type']=='fill' and x['side']=='sell'];self.assertEqual(len(sells),2);self.assertTrue(all(x['price']<900 and x['at']==o['observed_ms'] for x in sells))
    def test_compounding_and_fee_accounting(self):
        for exitprice in [800.,1200.]:
            s=core.initial(T)
            fills=[]
            for k in range(7):
                o=observation(T+k*I,entry=k<5,exit=k>=5,price=1000 if k<5 else exitprice)
                s,e=core.advance(s,o['signals'],o['books'],RULES,o['slot'],o['observed_ms']);fills += [x for x in e if x['type']=='fill']
            expected=10000-sum((1 if x['side']=='buy' else -1)*x['qty']*x['price']+x['fee'] for x in fills)
            self.assertAlmostEqual(s['cash'],expected);self.assertEqual(s['round_trips'],2)
            self.assertEqual(s['cash']>10000,exitprice>1000)
            for k in range(103,108):
                o=observation(T+k*I);s,e=core.advance(s,o['signals'],o['books'],RULES,o['slot'],o['observed_ms'])
                for x in e:
                    if x['type']=='fill' and x['side']=='buy':self.assertAlmostEqual(x['pre_entry_equity']*.125,x['qty']*x['price']+x['fee'],delta=.2)
    def test_future_data_cannot_change_signals(self):
        n=96*70;t=np.arange(n)*I;rng=np.random.default_rng(44);price=1000*np.exp(np.cumsum(rng.normal(0,.001,n)))
        a=np.column_stack([price,price*1.001,price*.999,price,np.ones(n),t]);cut=96*67
        b=a.copy();b[cut:,:4]*=2;b[cut:,4]*=100
        f,p=core.compute_signals(a);g,q=core.compute_signals(b)
        for key in ['p','mom30','d_atr','h4_atr','valid']:
            np.testing.assert_allclose(f[key][:cut+1],g[key][:cut+1],equal_nan=True)
        for key in ['entry','exit','gate']:np.testing.assert_array_equal(p[key][:cut+1],q[key][:cut+1])
    def test_partial_stale_and_crossed_inputs_rejected(self):
        s=runner.initial(T,{})
        for mode in ['partial','stale','crossed','slot']:
            o=observation(T)
            if mode=='partial':del o['books']['sol_jpy']
            if mode=='stale':o['books']['sol_jpy']['timestamp']-=I
            if mode=='crossed':o['books']['sol_jpy']['ask']=1
            if mode=='slot':o['signals']['sol_jpy']['slot']+=I
            with self.assertRaises(ValueError):runner.apply_tick(s,o,RULES)
        self.assertEqual(s['last_slot'],None)
    def test_journal_replay_restart_idempotence_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'comparison.sqlite3';db=runner.connect(path,True);s=runner.initialize(db,T,{})
            with self.assertRaises(RuntimeError):runner.initialize(db,T+I,{})
            for k in range(9):
                o=observation(T+k*I,exit=k>=6);s,events,_=runner.apply_tick(s,o,RULES);runner.commit(db,s,{'type':'tick','observation':o,'events':events})
            self.assertTrue(runner.verify(db,RULES,{})['ok']);db.close();db=runner.connect(path)
            same,ev,_=runner.apply_tick(runner.load(db),o,RULES);self.assertEqual(same,s);self.assertFalse(ev)
            corrupt=copy.deepcopy(s);corrupt['multi']['base']['cash']+=10
            with db:db.execute('UPDATE state SET payload=?',(runner.enc(corrupt),))
            with self.assertRaises(ValueError):runner.verify(db,RULES,{})
    def test_error_records_reconcile(self):
        with tempfile.TemporaryDirectory() as d:
            db=runner.connect(Path(d)/'comparison.sqlite3',True);s=runner.initialize(db,T,{})
            s=runner.error_state(s,'timeout');runner.commit(db,s,{'type':'error','at':T,'message':'timeout'})
            self.assertTrue(runner.verify(db,RULES,{})['ok']);self.assertEqual(s['multi']['base']['cash'],10000)
    def test_no_buy_when_exposure_drifts_above_cap(self):
        s=core.initial(T)
        for k in range(5):
            o=observation(T+k*I);s,_=core.advance(s,o['signals'],o['books'],RULES,o['slot'],o['observed_ms'])
        o=observation(T+5*I,price=10000);s,e=core.advance(s,o['signals'],o['books'],RULES,o['slot'],o['observed_ms'])
        self.assertGreater(s['exposure'],.25);self.assertFalse(any(x['type']=='fill' for x in e))
    def test_quote_validation(self):
        obj={'data':{'asks':[['1001','2']],'bids':[['999','2']],'timestamp':T}}
        self.assertEqual(runner.validate_book(obj,{'data':{'mode':'NONE'}},T)['mid'],1000)
        with self.assertRaises(ValueError):runner.validate_book(obj,{'data':{'mode':'NONE'}},T+I)
        with self.assertRaises(ValueError):runner.validate_book(obj,{'data':{'mode':'HALT'}},T)
if __name__=='__main__':unittest.main(verbosity=2)
