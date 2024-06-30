include("Aux.jl")
include("Protocols.jl")


#this file contains data structures and functions used by the main code 

"""
`edm!(du,u,param,t;learning_rule=unbounded_smith) -> Array{Float}`

total dynamics of our model, uses unbounded_smith(p,u,t,i;λ=0.1,τ=0.1) as the protocol by default
""" 
function edm_memoryless_payoff!(dx,x,param::Game,t;learning_rule=unbounded_smith)


    normalize!(x,1)
    x .= abs.(x) #for any pop game this is ok

    p = param.F(param, x, 0.0) #payoff

    dx .= learning_rule(x,p)


    nothing
end


"""
`edm!(du,u,param,t;learning_rule=unbounded_smith) -> Array{Float}`

total dynamics of our model, uses unbounded_smith(p,u,t,i;λ=0.1,τ=0.1) as the protocol by default
""" 
function edm_dynamic_pdm!(du,u,param,t;learning_rule=unbounded_smith)

    ns   = param.ns
    nq   = param.nq
    qdot = param.q_dot
    F    = param.F
    
    x = u[1:ns]
    q = u[(1:nq).+ns]


    normalize!(x,1)
    u[1:ns] .= abs.(x) #for any pop game this is ok

    p = F(param, x, q, 0.0) #payoff

    dx = learning_rule(x,p)
    dq = qdot(param,x,q,0.0)  

    du .= [dx;dq]
    
    nothing
end


