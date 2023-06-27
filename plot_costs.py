import math

def calc_pnl(k, position, price_change):
    p0 = 1
    x = math.sqrt(k) - position
    y = k/(math.sqrt(k) - position)
    p1 = y/x
    p2 = p1 * (1 + price_change)

    return math.sqrt(k) * (p2/p1 + math.sqrt(p1/p0) - p2/math.sqrt(p1 * p0) - 1)

#print(calc_pnl(1, 0.1, 0.01))
#print(calc_pnl(1, 0.1, 0.1))
#print(calc_pnl(1, 0.1, 1))


print(calc_pnl(1, 0.1, 0.01)/0.1)

for k in range(1, 100):
    position = 0.1 * math.sqrt(k)
    print(calc_pnl(k * 1.0, position, 0.01) / position)

'''
print(calc_pnl(1, 0.2, 0.01)/0.2)
print(calc_pnl(1, 0.3, 0.01)/0.3)
print(calc_pnl(1, 0.4, 0.01)/0.4)
print(calc_pnl(1, 0.5, 0.01)/0.5)
print(calc_pnl(1, 0.6, 0.01)/0.6)
print(calc_pnl(1, 0.7, 0.01)/0.7)
print(calc_pnl(1, 0.8, 0.01)/0.8)
print(calc_pnl(1, 0.9, 0.01)/0.9)
'''
#print(calc_pnl(1, 0.2, 0.1))
#print(calc_pnl(1, 0.2, 1))