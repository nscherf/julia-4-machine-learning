using Learn
using MLDataUtils
using StatsBase
using StatPlots
import MNIST

ENV["GKS_WSTYPE"] = "x11"
gr(leg=false, linealpha = 0.5)

type TracePlot{I,T}
  indices::I
  plt::Plot{T}
end

getplt(tp::TracePlot) = tp.plt

function TracePlot(n::Int = 1; maxn::Int = 500, kw...)
  indices = if n > maxn
    shuffle(1:n)[1:maxn]
  else
    1:n
  end
  TracePlot(indices, plot(length(indices); kw...))
end

function add_data(tp::TracePlot, x::Number, y::AbstractVector)
  for (i,idx) in enumerate(tp.indices)
    push!(tp.plt.series_list[i], x, y[idx])
  end
end

add_data(tp::TracePlot, x::Number, y::Number) = add_data(tp,x,[y])

x_train, y_train = MNIST.traindata()
x_test, y_test = MNIST.testdata()

mu, sigma = rescale!(x_train)
rescale!(x_test, mu, sigma)

y_train, y_test = map(to_one_hot, (y_train, y_test))

train = (x_train, y_train)
test = (x_test, y_test)

nin, nh, nout = 784, [50,50], 10
t = nnet(nin, nout, nh, :softplus, :softmax)

obj = objective(t, ElasticNetPenalty(1e-5))

pidx = 1:2:length(t)
pvalplts = [TracePlor(length(params(t[i])), title="$(t[i])") for i=pidx]
