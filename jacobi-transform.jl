function convergence(A::Array)
  num = 0.0
  l = size(A)[1]
  for ii in 1:(l-1)
    for jj in (ii+1):l
      num+=A[ii,jj]^2
    end
  end
  return num
end

  function roundmatrix(A::Array, rtol::Real)
    Ap=copy(A)
    for ii in 1:length(A)
      if abs(Ap[ii])<rtol
        Ap[ii]=0
      end
    end
    return Ap;
  end

  function makeA(n::Int)
    A=randn(n,n);
    for ii in 1:n
      A[ii,1:ii]=transpose(A[1:ii,ii])
    end
    V = eye(n)
    return A, copy(A), V
  end

function Rotate(A::Array, p::Int, q::Int; computeV=false, V::Array=eye(1))
  theta = (A[q,q]-A[p,p])/(2*A[p,q]);
  t = sign(theta) / (abs(theta)+sqrt(theta^2+1));

  c = 1/sqrt(t^2+1)
  s = t*c
  tau = s/(1+c)

  l = size(A)[1]
  Ap = copy(A[:,p])
  Aq = copy(A[:,q])
  for r in 1:l
    A[r,p]=Ap[r]-s*(Aq[r]+tau*Ap[r])
    A[r,q]=Aq[r]+s*(Ap[r]-tau*Aq[r])

    A[p,r]=A[r,p]
    A[q,r]=A[r,q]
  end
  A[p,q]=0
  A[q,p]=0
  A[p,p]=Ap[p]-t*Aq[p]
  A[q,q]=Aq[q]+t*Aq[p]

  if computeV==true
    Vp=copy(V[:,p])
    Vq=copy(V[:,q])
    for r in 1:l
      V[r,p]=c*Vp[r]-s*Vq[r]
      V[r,q]=s*Vp[r]+c*Vq[r]
    end
    return A,V
  else
    return A;
  end
end
