
import math
import numpy as np
import graph
import calc_std

class uni:
    def __init__(self, funding_rate_num = 1, funding_rate_den = 24, no_reppeg = False):
        self.x = 1
        self.y = 1

        self.total_y_traders_hold = 0
        self.total_x_traders_hold = 0

        self.acc_funding_rate = 0
        self.funding_rate_den = funding_rate_den
        self.funding_rate_num = funding_rate_num

        self.no_reppeg = no_reppeg

        self.x0 = 1
        self.y0 = 1

    def clone(self):
        temp = uni()
        temp.x = self.x
        temp.y = self.y

        temp.total_y_traders_hold = self.total_y_traders_hold
        temp.total_x_traders_hold = self.total_x_traders_hold

        temp.acc_funding_rate = self.acc_funding_rate

        return temp

    def tick(self, mark_price, do_trade = True):
        x = self.x
        y = self.y
        curr_price = y/x

        if curr_price > mark_price:
            # price went down. someone dumped x, and got y
            # (x + dx)(y - dy) = xy
            # (y - dy) = mark * (x + dx)

            # (x + dx)**2 = xy/mark
            # dx = sqrt(xy/mark) - x

            dx = math.sqrt(x*y / mark_price) - x

            # dy = y - (mark * (x + dx))
            dy = y - mark_price * (x + dx)

            if do_trade:
                self.total_y_traders_hold += dy
                self.total_x_traders_hold -= dx

            self.x += dx
            self.y -= dy
        else:
            # price went up. someone dumped y and got x
            # (x - dx)(y + dy) = xy
            # (y + dy) = mark * (x - dx)

            # (x - dx)**2 = xy/mark
            # dx = x - sqrt(xy/mark)

            dx = x - math.sqrt(x*y / mark_price)

            # dy = mark * (x - dx) - y
            dy = mark_price * (x - dx) - y

            if do_trade:
                self.total_x_traders_hold += dx
                self.total_y_traders_hold -= dy

            self.y += dy
            self.x -= dx
        
        #print(round(self.y / self.x, 5), round(mark_price, 5))
        #assert (round(self.y / self.x, 5) == round(mark_price, 5))

    def pnl_for_reppeg(self, reppeg_to):
        p0 = self.y0 / self.x0
        p1 = self.y / self.x
        p2 = reppeg_to

        return self.y0 * (p2/p1 + math.sqrt(p1/p0) - p2/math.sqrt(p1 * p0) - 1)

    def pnl_after_repeg(self, mark_price):
        temp = self.clone()
        pnl_before = temp.pnl_in_x()
        temp.tick(mark_price, False)
        pnl_after = temp.pnl_in_x()

        #print("ratio", (pnl_after - pnl_before) / self.pnl_for_reppeg(mark_price), (pnl_after - pnl_before), self.pnl_for_reppeg(mark_price))
        #print("x0", self.x0, "y0", self.y0, "y1", self.y, "x1", self.x, "reppeg to", mark_price, "x held by traders", self.total_x_traders_hold,
        #      "y held by traders", self.total_y_traders_hold)

        return pnl_after

    def repeg(self, mark_price):
        if self.acc_funding_rate + self.pnl_after_repeg(mark_price) >= 0 and (not self.no_reppeg):
            self.tick(mark_price, False)

            self.x0 = abs(self.x + self.total_x_traders_hold)
            self.y0 = self.x * self.y / self.x0

            return True

        return False

    

    def pnl_in_x(self):
        x = self.x
        y = self.y

        if self.total_x_traders_hold < 0:
            # dump all y assets
            dy = self.total_y_traders_hold
            # (y+dy)(x-dx) = xy
            # dx = x - xy/(y+dy)

            dx = x - x*y/(y+dy)
            return -(self.total_x_traders_hold + dx)

        else:
            # buy enough y assets to cover depicit
            dy = -self.total_y_traders_hold

            # (y-dy)(x+dx) = xy
            # dx = xy/(y-dy) - x
            dx = x*y/(y-dy) - x

            return (dx - self.total_x_traders_hold)

    def calc_total_long_minus_short(self):
        return self.total_x_traders_hold

        mark_price = self.y / self.x

        # 1) calc x shorts
        dy = self.total_y_traders_hold
        x = self.x
        y = self.y
        # (y+dy)(x-dx) = xy
        # dx = x - xy/(y+dy)
        dx = x - x * y / (y + dy)
        x_short = dx
        x_long = self.total_x_traders_hold

        return {"long" : x_long, "short" : x_short}

    def usual_funding_rates(self, index_price):
        mark_price = self.y / self.x
        return (abs(mark_price - index_price) / index_price) * abs(self.total_x_traders_hold/math.sqrt(2)) / (24)

    def collect_extra_funding_rate(self, index_price):
        mark_price = self.y / self.x

        curr_pnl = self.pnl_in_x()
        pnl_after_reppeg = self.pnl_after_repeg(index_price)

        reppeg_cost = -(pnl_after_reppeg - curr_pnl)

        normal_rates = self.usual_funding_rates(index_price)

        # initially, just add to pnl, TODO - come up with a strategy
        
        if reppeg_cost > 0:
            actual_costs = reppeg_cost * self.funding_rate_num / self.funding_rate_den # have 24 hours till reppeg on average

            five_percent_costs = reppeg_cost * 0.05 / (abs(mark_price - index_price) / index_price )

            if normal_rates < actual_costs:
                actual_costs = normal_rates

            if self.acc_funding_rate < five_percent_costs:
                self.acc_funding_rate += actual_costs
            self.acc_funding_rate += actual_costs
        #else:
        #    self.acc_funding_rate += normal_rates * 0.5
        #'''
        #self.acc_funding_rate += normal_rates * 0.2
        return {"reppeg_cost" : reppeg_cost, "normal_funding" : normal_rates}




        
def up_only():
    u = uni()
    mark_price = 0
    for i in range(10):
        mark_price = 1 + i / 100
        print("pnl after repeg", u.pnl_after_repeg(mark_price))
        u.collect_extra_funding_rate(mark_price)
        print("acc funding rates", u.acc_funding_rate)        
        u.tick(mark_price)
        print(u.x, u.y, u.x * u.y, u.y/u.x, mark_price, u.calc_total_long_minus_short())
        print("pnl", u.pnl_in_x())

    for i in range(10):
        mark_price = mark_price / 1.01
        print("pnl after repeg", u.pnl_after_repeg(mark_price))
        u.collect_extra_funding_rate(mark_price)
        print("acc funding rates", u.acc_funding_rate)        
        u.tick(mark_price)
        print(u.x, u.y, u.x * u.y, u.y/u.x, mark_price, u.calc_total_long_minus_short())
        print("pnl", u.pnl_in_x())

def defi_smart(num_iterations, num, den, seed, no_reppeg, with_logs):
    u = uni(num, den, no_reppeg)
    seed_value = seed
    np.random.seed(seed_value)
    #random_array = np.random.uniform(-0.01, 0.01, num_iterations)
    # avg price change (in abs values) is 0.002
    random_array = np.random.normal(0, 0.002 * math.sqrt(3.14159/2), num_iterations)
    mark_price = 1.0
    index_price = 1.0

    historical_mark = []
    historical_index = []

    cntr = 0

    string = "infinite"
    if not no_reppeg:
        string = str(den/num)

    file_name = "reppeg_every_" + string + "_hours.csv"
    file_name = "acc_funds_only_up_to_5_percent.csv"

    if with_logs:
        f = open(file_name, "w")
        print("block_time,index_price,mark_price,acc_funding,pnl_in_x,long_minus_short,reppeg,reppeg_cost,x_hold,y_hold,x,y,reppeg_costs,normal_funding", file = f)

    for price_delta in random_array:
        # try to reppeg
        reppeg_result = False
        
        pnl_before = u.pnl_in_x()
        if index_price > mark_price * 1.001:
            reppeg_result = u.repeg(mark_price * 1.001)
        elif index_price * 1.001 < mark_price:
            reppeg_result = u.repeg(mark_price / 1.001)
        pnl_after = u.pnl_in_x()

        pnl_cost = pnl_after - pnl_before

        if reppeg_result:
            reppeg_result = 1
        else:
            reppeg_result = 0
        #print(price_delta)

        mark_delta = np.random.normal(price_delta, 0.001)
        #print(mark_delta)

        mark_price = u.y / u.x

        '''
        if index_price >= mark_price and index_price < mark_price * 1.05:
            if price_delta < mark_delta:
                mark_delta = price_delta

        if mark_price > index_price and mark_price < index_price * 1.05:
            # index < mark
            if price_delta > mark_delta:
                mark_delta = price_delta
        '''
        #print(index_price, mark_price, price_delta, mark_delta)
        #print(mark_delta, price_delta)
        index_price *= math.exp(price_delta)
        mark_price *= math.exp(mark_delta)
        
        '''
        index_price *= (1 + price_delta)
        mark_price *= (1 + mark_delta)
        '''
        cntr += 1

        u.tick(mark_price)
        fund_result = u.collect_extra_funding_rate(index_price)

        historical_index.append(index_price)
        historical_mark.append(mark_price)

        if with_logs:
            print(cntr, ",", index_price, ",", mark_price, ",", u.acc_funding_rate, ",", u.pnl_in_x(), ",", u.calc_total_long_minus_short(), ",", reppeg_result, ",",
                  pnl_cost, ",", u.total_x_traders_hold, ",", u.total_y_traders_hold, ",", u.x, ",", u.y, ",",
                  fund_result["reppeg_cost"], ",", fund_result["normal_funding"], file = f)

    if with_logs:
        f.close()

        graph.read_csv_file(file_name)
        calc_std.read_csv_file(file_name)

    # calc R2
    r2 = graph.calculate_r2(np.array(historical_index), np.array(historical_mark))
    return {"r2" : r2, "pnl" : u.pnl_in_x(), "acc_funding" : u.acc_funding_rate, "long_minus_short" : u.calc_total_long_minus_short()}

#up_only(100)
'''
for mult in range(5):
    r2_results = []
    funding_results = []
    long_results = []
    pnl = []
    for i in range(1000):
        res = defi_smart(10000, 1, 24, i, False, False)
        print(i, res)
        r2_results.append(res["r2"])
        funding_results.append(res["acc_funding"])
        long_results.append(res["long_minus_short"])
        pnl.append(res["pnl"])

    print("avg r2", np.mean(r2_results), "r2 std", np.std(r2_results))
    print("avg acc funding", np.mean(funding_results), "acc funding std", np.std(funding_results))
    print("avg long minus short", np.mean(long_results), "median long minus short", np.median(long_results))
    print("avg pnl", np.mean(pnl), "median", np.median(pnl), "std", np.std(pnl))
    r2_value = np.percentile(r2_results, 10)
    print(r2_value)
    print("r2 percentlie", r2_value, "index", np.where(np.isclose(r2_results, r2_value, 1e-3))[0][0])
    elem_index = np.where(np.isclose(r2_results, r2_value, 1e-3))[0][0]
    defi_smart(10000, 1, 24 * 10, elem_index, False, True)
    break
#np.where(np.isclose(data, value, atol=tolerance))
'''
res = defi_smart(10000, 1, 24, 0, False, True)
print(res)
