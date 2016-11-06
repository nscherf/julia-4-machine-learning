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
pvalplts = [TracePlot(length(params(t[i])), title="$(t[i])") for i=pidx]
ylabel!(pvalplts[1].plt, "parameter vals")
pgradplts = [TracePlot(length(params(t[i]))) for i=pidx]
ylabel!(pgradplts[1].plt, "parameter grads")

valinplts = [TracePlot(input_length(t[i]),title="input", yguide="Layer value") for i=1:1]
valoutplts = [TracePlot(output_length(t[i]), title="$(t[i])", titlepos=:left) for i=1:length(t)]
gradinplts = [TracePlot(input_length(t[i]), yguide="Layer Grad") for i=1:1]
gradoutplts = [TracePlot(output_length(t[i])) for i=1:length(t)]

# loss/accuracy plots
lossplt = TracePlot(title="Test Loss", ylim=(0,Inf))
accuracyplt = TracePlot(title="Accuracy", ylim=(0.6,1))

function my_test_loss(obj, testdata, totcount = 500)
    totloss = 0.0
    totcorrect = 0
    for (x,y) in eachobs(rand(eachobs(testdata), totcount))
        totloss += transform!(obj,y,x)

        # logistic version:
        # ŷ = output_value(obj.transformation)[1]
        # correct = (ŷ > 0.5 && y > 0.5) || (ŷ <= 0.5 && y < 0.5)

        # softmax version:
        ŷ = output_value(obj.transformation)
        chosen_idx = indmax(ŷ)
        correct = y[chosen_idx] > 0

        totcorrect += correct
    end
    totloss, totcorrect/totcount
end
tracer = IterFunction((obj, i) -> begin
    n = 100
    mod1(i,n)==n || return false

    # param trace
    for (j,k) in enumerate(pidx)
        add_data(pvalplts[j], i, params(t[k]))
        add_data(pgradplts[j], i, grad(t[k]))
    end

    # input/output trace
    for j=1:length(t)
        if j==1
            add_data(valinplts[j], i, input_value(t[j]))
            add_data(gradinplts[j], i, input_grad(t[j]))
        end
        add_data(valoutplts[j], i, output_value(t[j]))
        add_data(gradoutplts[j], i, output_grad(t[j]))
    end

    # compute approximate test loss and trace it
    if mod1(i,500)==500
        totloss, accuracy = my_test_loss(obj, test, 500)
        add_data(lossplt, i, totloss)
        add_data(accuracyplt, i, accuracy)
    end

    # build a heatmap of the total outgoing weight from each pixel
    pixel_importance = reshape(sum(t[1].params.views[1],1), 28, 28)
    hmplt = heatmap(pixel_importance, ratio=1)

    # build a nested-grid layout for all the trace plots
    plot(
        map(getplt, vcat(
                pvalplts, pgradplts,
                valinplts, valoutplts,
                gradinplts, gradoutplts,
                lossplt, accuracyplt
            ))...,
        hmplt,
        size = (1400,1000),
        layout=@layout([
            grid(2,length(pvalplts))
            grid(2,length(valoutplts)+1)
            grid(1,3){0.2h}
        ])
    )

    # show the plot
    gui()
end)

# trace once before we start learning to see initial values
tracer.f(obj, 0)
