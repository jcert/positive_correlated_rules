


#%%

#using Revise
using Pkg


#Pkg.add(["DifferentialEquations","ForwardDiff","JuMP","Ipopt","LaTeXStrings"])
using DifferentialEquations, LinearAlgebra
using ForwardDiff, JuMP, Ipopt, LaTeXStrings
using Plots



#USE_TIKZ = true 

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
    default(dpi=500)
end



if @isdefined DEF_CONFIG
    plt_size = (800,600)
    default(;DEF_CONFIG...)
else
    plt_size = (600,500)
    #plt_size = (900,900)
    #default(linewidth = 3, markersize=10, margin = 10*Plots.mm,
    #    tickfontsize=12, guidefontsize=20, legend_font_pointsize=18)

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







####### potential game


F(sd, x, t) = -(x-ones(3))  #payoff function (sd::SystemDescription,state::Vector{Vector{Real}},t::Real)


my_game = Game(ns,F)

## we can select a single learning_rule to be used by the population 
f!(dx,x,param,t) = edm_memoryless_payoff!(dx,x,param,t;learning_rule=bnn)
f!(dx,x,param,t) = edm_memoryless_payoff!(dx,x,param,t;learning_rule=unbounded_smith)

#or do hybrid_learning_rules, where the vector given to hybrid_rule_maker weighs the rules to be used 
my_hybrid_learning_rule = hybrid_rule_maker([0.3,0.3,0.3])
## another way of using hybrid_rule_maker is to explicitly list the rules being weighed as follows:
# my_hybrid_learning_rule = hybrid_rule_maker([0.001,0.1],learning_rules=[unbounded_smith,unbounded_smith])

#%%

##########
#here we plot trajectories for pure BNN and Smith
##########

##setup a clean plots
f1 = plot() 
f2 = plot() 
f3 = plot() 

sols = []
alphas_list = []
for x0 in X0
    for alphas in [[1.0;0.0;0.0],[0.0;1.0;0.0],[0.0;0.01;0.1]]
        T  = 100.0

        local my_hybrid_learning_rule = hybrid_rule_maker(alphas)
        
        # f! is the EDM
        f!(dx,x,param,t) = edm_memoryless_payoff!(dx,x,param,t;learning_rule=my_hybrid_learning_rule)


        prob = ODEProblem(f!,x0,[0.0,T],my_game)
        sol = solve(prob, alg_hints=[:stiff])

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
simplex_quiver_plot!(sols, FF, ptype=plot!, linescolor=aux_c)
plot!([1 1 1],[1 1 1],label=["BNN" "Smith" "??"], color=aux_c', lw=5)
annotate!([0; 1.03; -1].*1.1,[1; -1/2 ;-1/2].*1.1, ["1", "2", "3"])
xlims!(-1.3,1.3)
ylims!(-0.6,1.15)



title!("Canonical Learning Rules")
#%%
savefig("potential_quiver_pure."*plt_ext)

plot(sols[1])
savefig("potential_one_solution."*plt_ext)

#%%





####### example 4
 
my_k = -1.0
my_λ = 5.0
my_b = [2;0;0]/(my_k*my_λ)
my_A = [0 0 0; 0 1 0; 0 0 1]
my_A = (my_A+my_A')/2

q_dot(sd, x, q, t; λ=my_λ, A=my_A, b=my_b, k=my_k) = λ*(A*x+b-q)
mathcal_F(sd, x, t) = -(x-ones(3))
F(sd, x, q, t; λ=my_λ, A=my_A, b=my_b, k=my_k) = mathcal_F(sd, x, t) + k*λ*(A*x+b-q)


my_game = PDM(ns,ns,q_dot,F)


##########
#here we plot trajectories for hybrid, BNN + Smith + replicator
##########

##setup a clean plots
f1 = plot() 
f2 = plot() 
f3 = plot() 

sols = []
alphas_list = []
for x0 in X0
    for alphas in [[1;0;0],[0;1;0],[0.01;0;1]]
        T  = 100.0

        local my_hybrid_learning_rule = hybrid_rule_maker(alphas)

        # f! is the EDM+PDM
        f!(du,u,param,t) = edm_dynamic_pdm!(du,u,param,t;learning_rule=my_hybrid_learning_rule)

        prob = ODEProblem(f!,[x0;zero(x0)],[0.0,T],my_game)
        sol = solve(prob, alg_hints=[:stiff])

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
aux_ls = [:solid; :dot; :dashdotdot]

simplex_sols_plot!(sols, ptype=plot!, linescolor=aux_c, linestyles=aux_ls )

aux = 'c'
for x0 in X0
    global aux
    v0 = [0 1 -1;1 -1/2 -1/2]*x0

    x1 = copy(x0)
    x1[findmin(x0)[2]] -= 0.1
    v1 = [0 1 -1;1 -1/2 -1/2]*x1

    #v1 = v0.*(1+0.1/norm(v0,1))
    #v1 = rotation_mat2D(-5)*v1

    aux_c_local = palette(:Set1_5)[1]

    plot!([v0[1]], [v0[2]], ms=5,shape=:rect,c=aux_c_local, annotations = (v1[1], v1[2], Plots.text(aux, :right, aux_c_local)), label = nothing)
    aux = aux+1
end

#plot!([1 1 1],[1 1 1],label=[L"\mathcal{T}^\text{\tiny BNN}" L"\mathcal{T}^\text{\tiny Smith}" L"\mathcal{T}^\text{\tiny A}"], color=aux_c', ls=permutedims(aux_ls), lw=2)
plot!([1 1 1],[1 1 1],label=[L"\mathcal{T}^{BNN}" L"\mathcal{T}^{Smith}" L"\mathcal{T}^A"], color=aux_c', ls=permutedims(aux_ls), lw=2)
annotate!([0; 1.03; -1].*1.1,[1; -1/2 ;-1/2].*1.1, ["1", "2", "3"])
xlims!(-1.3,1.3)
ylims!(-0.6,1.15)


#title!("Rule = Replicator + 0.01*Smith ")
#%%
savefig("pdm."*plt_ext)
#%%
