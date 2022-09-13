def matvecxi(xi, parxi, AATmap):
    IpsigAATxi = xi + parxi.sigma * feval(AATmap, xi)
    return IpsigAATxi
