import math
k_1=float
k_0=float
n=float
i=float
mm=float
r=float
nom_int=float
inf_rate=float
real_int=float
def Compound(k_0,n,i):
    k_0=input("Initial Capital:")
    k_0=int(k_0)
    n=input("Number of periods/Time:")
    n=int(n)
    i=input("Interest rate:")
    i=float(i)
    return k_0*(i+1)**n
def InterestRate(k_0,k_1,n):
    k_0=input("Initial Capital:")
    k_0=float(k_0)
    k_1=input("Final Capital:")
    k_1=float(k_1)
    n=input("Number of periods/Time:")
    n=float(n)
    return 100*(k_1/k_0)**(1/n)
def ReserveRatio(mm):
    mm=input("Money Multiplier:")
    mm=float(mm)
    return 1/mm
def MoneyMultiplier(r):
    r=input("Reserve Ratio:")
    r=float(r)
    return 1/r
def RealInterestRate(nom_int,inf_rate):
    nom_int=input("Nominal Interest Rate:")
    inf_rate=input("Inflation Rate:")
    nom_int=float(nom_int)
    inf_rate=float(inf_rate)
    return nom_int-inf_rate
def NominalInterestRate(real_int,inf_rate):
    real_int=input("Real Interest Rate:")
    inf_rate=input("Inflation Rate:")
    real_int=float(real_int)
    inf_rate=float(inf_rate)
    return real_int+inf_rate
function=input("Function:")
function=function.lower()
function=function.replace(" ","")
if function == "compound":
        print("Final Capital $",round(Compound(k_0,n,i)))
if function == "interestrate":
        print("Interest Rate(Percent):",round(InterestRate(k_0,k_1,n)))
if function == "reserveratio":
        print("Reserve Ratio",(ReserveRatio(mm)))
if function == "moneymultiplier":
        print("Money Multiplier:",MoneyMultiplier(r))
if function == "realinterestrate":
        print("Real Interest Rate:",RealInterestRate(nom_int,inf_rate))
if function == "nominalinterestrate":
        print("Nominal Interest Rate:",NominalInterestRate(real_int,inf_rate))
        
