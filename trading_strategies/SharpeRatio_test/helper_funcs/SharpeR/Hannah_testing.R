load("Sharpe.RData")


hac.inference(ret.hedge)

boot.time.inference(ret = ret.hedge,b = 5, M = 1000)
