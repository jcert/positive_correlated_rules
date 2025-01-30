


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
    Plots.scalefontsizes(1.7) #this can cause issues when running the code multiple times
    default(dpi=500)
end



if @isdefined DEF_CONFIG
    plt_size = (800,600)
    default(;DEF_CONFIG...)
else
    plt_size = (600,900)
    #plt_size = (900,900)
    default(linewidth = 3, markersize=10, margin = 10*Plots.mm)

end

include("Aux.jl")
include("Protocols.jl")
include("Dynamics.jl")



# number of strategies
ns=3               

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

F(sd, x, dx3, hdx3, t) = -[ 4+3; 4*exp(2*x[2]-1)+4*(x[2]+x[3]); 4+4*(x[2]+x[3])+(dx3+hdx3) ] #ccw payoff


TT = 100.0
x0 = [1;0;0]
my_game = (ns=3, tau=4, F=F)

p_hist = []

function pwd_model!(du, u, hu, p, t; ϵ=0.001, learning_rule=unbounded_smith)
    pdm = p
    my_ns, my_tau, my_F = pdm

    dx3  = (u[3]- hu(p,t-ϵ)[3])/ϵ #approximate the derivative of the current term
    hdx3 = (hu(p,t-my_tau)[3]- hu(p,t-my_tau-ϵ)[3])/ϵ #approximate the derivative of the delay term
    
    ns   = my_ns
    
    x = u[1:ns]

    normalize!(x,1)
    u[1:ns] .= abs.(x) #for any pop game this is ok

    p = my_F(pdm, x, dx3, hdx3, t) #payoff

    append!(p_hist,p)

    dx = learning_rule(x,p)

    du .= dx
    
    nothing

end



## we can select a single learning_rule to be used by the population 
f!(dx,x,hx,param,t) = pwd_model!(dx, x, hx, param, t; learning_rule=bnn)
f!(dx,x,hx,param,t) = pwd_model!(dx, x, hu, param, t; learning_rule=unbounded_smith)

#or do hybrid_learning_rules, where the vector given to hybrid_rule_maker weighs the rules to be used 
my_hybrid_learning_rule = hybrid_rule_maker([0.3,0.3,0.3])
## another way of using hybrid_rule_maker is to explicitly list the rules being weighed as follows:
# my_hybrid_learning_rule = hybrid_rule_maker([0.001,0.1],learning_rules=[unbounded_smith,unbounded_smith])

#%%

##setup a clean plots
f1 = plot() 
f2 = plot() 
f3 = plot() 

sols = []
alphas_list = []
for x0 in X0
    for alphas in [[1;0;0],[0;1;0],[0;0.01;1]]
        T  = TT

        local my_hybrid_learning_rule = hybrid_rule_maker(alphas)

        # f! is the EDM
        f!(dx,x,hx,param,t) = pwd_model!(dx, x, hx, param, t; learning_rule=my_hybrid_learning_rule)

        prob = DDEProblem(f!, x0, (p,t)->zeros(3), [0.0,T], my_game)

        # -[ 2+1.4; 2*exp(x[2])+2*(x[2]+x[3]); 2+2*(x[2]+x[3])+dx3+hdx3 ] #ccw payoff

        #alg = MethodOfSteps(RK4())
        alg = MethodOfSteps(Tsit5())
        sol = solve(prob, alg, saveat = T/1000, alg_hints=[:stiff])

        push!(sols, sol)
        push!(alphas_list, alphas)

    end
end


#plot a new trajectory on the plot
f1 = plot()

FF(x) = my_game.F(my_game, x, 0.0)
#[:red :blue :green]) 
# palette(:Set1_5)
aux_c = palette(:Set1_5)[[3,5,2]]
#aux_c = palette(:Set1_5)[[3,5]]

# :auto, :solid, :dash, :dot, :dashdot, :dashdotdot]
aux_ls = [:solid; :dot; :dashdot]

simplex_sols_plot!(sols, ptype=plot!, linescolor=aux_c, linestyles=aux_ls )

aux = 'c'
for x0 in X0
    global aux
    v0 = [0 1 -1;1 -1/2 -1/2]*x0

    x1 = copy(x0)
    x1[findmin(x0)[2]] -= 0.15
    v1 = [0 1 -1;1 -1/2 -1/2]*x1

    #v1 = v0.*(1+0.1/norm(v0,1))
    #v1 = rotation_mat2D(-5)*v1

    aux_c_local = palette(:Set1_5)[1]

    plot!([v0[1]], [v0[2]], ms=5,shape=:rect,c=aux_c_local, annotations = (v1[1], v1[2], (aux, aux_c_local)), label = nothing)
    aux = aux+1
end

#plot!([1 1 1],[1 1 1],label=[L"\mathcal{T}^\text{\tiny BNN}" L"\mathcal{T}^\text{\tiny Smith}" L"\mathcal{T}^\text{\tiny A}"], color=aux_c', ls=permutedims(aux_ls), lw=2)
plot!([1 1 1],[1 1 1],label=[L"\mathcal{T}^{BNN}" L"\mathcal{T}^{Smith}" L"\mathcal{T}^b"], color=aux_c', ls=permutedims(aux_ls), lw=2)
annotate!([0; 1.03; -1].*1.2,[1; -1/2 ;-1/2].*1.2, ["1", "2", "3"])
xlims!(-1.3,1.3)
ylims!(-0.7,1.3)


#title!("Rule = Replicator + 0.01*Smith ")
#%%
savefig("img/pdm."*plt_ext)
#%%
