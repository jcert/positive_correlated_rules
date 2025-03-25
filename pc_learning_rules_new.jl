


#%%

#using Revise
using Pkg


#Pkg.add(["DifferentialEquations","ForwardDiff","JuMP","Ipopt","LaTeXStrings"])
using DifferentialEquations, LinearAlgebra
using ForwardDiff, JuMP, Ipopt, LaTeXStrings
using Plots


if !isdir("img")
    mkpath("img") 
end

USE_TIKZ = true
#plotly()
gr()
   
USE_DEFED = @isdefined USE_TIKZ 
if USE_DEFED && (USE_TIKZ)
    pgfplotsx()
    plt_ext = "tikz"
else
    #plotly()
    gr()
    plt_ext = "png"
    Plots.scalefontsizes()
    Plots.scalefontsizes(1.7) #this can cause issues when running the code multiple times
    default(dpi=500)
end



if @isdefined DEF_CONFIG
    plt_size = (600,600)
    default(;DEF_CONFIG...)
else
    plt_size = (600,600)
    #plt_size = (900,900)
    default(linewidth = 3, markersize=10, margin = 10*Plots.mm)

end

include("Aux.jl")
include("Protocols.jl")
include("Dynamics.jl")



# number of strategies
ns=3
qs=1               

# initial conditions
X0 = [
    #normalize!([10.0;0.0;0.0],1),
    normalize!([0.0;10.0;0.0],1),
    #normalize!([0.0;0.0;10.0],1),
    normalize!([7.0;3.0;0.0],1),
    normalize!([0.0;2.0;8.0],1),
    normalize!([6.0;0.0;4.0],1),
    #normalize!([10.0;10.0;0.0],1),
    #normalize!([0.0;10.0;10.0],1),
    #normalize!([7.0;0.0;10.0],1)
    ]

#%%







####### CCW dynamic payoff


#payoff function (sd::SystemDescription,pop_state::Vector{Vector{Real}},pdm_state::Vector{Vector{Real}},t::Real)
F(sd, x, q, t) = -[ 5+6*(x[1]+x[3]); 16*x[2]^2+4; 6*(x[1]+x[3])+4*x[3]+1*(x[3]-q[1])+4 ] #ccw payoff v2.1


#pdm dynamics    (sd::SystemDescription,pop_state::Vector{Vector{Real}},pdm_state::Vector{Vector{Real}},t::Real)
my_q_dot(sd, x, q, t) = -q .+ x[3]

my_game = PDM(ns,qs,my_q_dot,F)

## we can select a single learning_rule to be used by the population 
f!(dx,x,param,t) = edm_memoryless_payoff!(dx,x,param,t;learning_rule=bnn)
f!(dx,x,param,t) = edm_memoryless_payoff!(dx,x,param,t;learning_rule=unbounded_smith)

#or do hybrid_learning_rules, where the vector given to hybrid_rule_maker weighs the rules to be used 
my_hybrid_learning_rule = hybrid_rule_maker([0.3,0.3,0.3])
## another way of using hybrid_rule_maker is to explicitly list the rules being weighed as follows:
# my_hybrid_learning_rule = hybrid_rule_maker([0.001,0.1],learning_rules=[unbounded_smith,unbounded_smith])


##setup a clean plots
f1 = plot() 
f2 = plot() 
f3 = plot() 

sols = []
alphas_list = []
for x0 in X0
    for alphas in [[1;0;0],[0;1;0],[0;0.01;1]]
        T  = 15.0

        local my_hybrid_learning_rule = hybrid_rule_maker(alphas)

        # f! is the EDM+PDM
        f!(du,u,param,t) = edm_dynamic_pdm!(du,u,param,t;learning_rule=my_hybrid_learning_rule)

        prob = ODEProblem(f!,[x0;zeros(qs)],[0.0,T],my_game)
        sol = solve(prob, alg_hints=[:stiff])

        push!(sols, sol)
        push!(alphas_list, alphas)

    end
end

#plot a new trajectory on the plot
f1 = plot()

aux_c = palette(:Set1_5)[[3,5,2]]

# :auto, :solid, :dash, :dot, :dashdot, :dashdotdot]
aux_ls = [:solid; :dot; :dashdot]

simplex_sols_plot!(sols, ptype=plot!, linescolor=aux_c, linestyles=aux_ls )

aux = 'c'
for x0 in X0
    global aux
    v0 = psim*x0

    x1 = copy(x0)
    x1[findmin(x0)[2]] -= 0.15
    v1 = psim*x1


    aux_c_local = palette(:Set1_5)[1]

    plot!([v0[1]], [v0[2]], ms=5,shape=:rect,c=aux_c_local, annotations = (v1[1], v1[2], (aux, 12, aux_c_local)), label = nothing)
    aux = aux+1
end

#plot!([1 1 1],[1 1 1],label=[L"\mathcal{T}^\text{\tiny BNN}" L"\mathcal{T}^\text{\tiny Smith}" L"\mathcal{T}^\text{\tiny A}"], color=aux_c', ls=permutedims(aux_ls), lw=2)
plot!([1 1 1],[1 1 1],label=[L"\mathcal{T}^{BNN}" L"\mathcal{T}^{Smith}" L"\mathcal{T}^b"], color=aux_c', ls=permutedims(aux_ls), lw=3)
#annotate!([0; 1.03; -1].*1.2,[1; -1/2 ;-1/2].*1.2, ["1", "2", "3"] )
p = psim*I(3)
annotate!(p[1,:].*1.1, p[2,:].*1.1, text.(["1", "2", "3"],12) )
xlims!(-0.8,0.8)
ylims!(-0.4,0.6)


#title!("Rule = Replicator + 0.01*Smith ")
#%%
savefig("img/pdm."*plt_ext)
#%%
