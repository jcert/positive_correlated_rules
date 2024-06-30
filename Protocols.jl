include("Aux.jl")

#this file contains data structures and functions used by the main code 



"""
`unbounded_smith(x::Vector{<:Real},p::Vector{<:Real}; λ=1.0, τ=1.0 ) -> Vector{<:Real}`

Smith dynamics. Takes parameters `x` (a vector with the current state of the population
and `p` the currently offered payoff vector.

# Examples
```julia
(TODO)
julia> unbounded_smith(ones(ns),zeros(ns); λ=7.0, τ=3.0 )
...
```
""" 
function unbounded_smith(x::Vector{<:Real},p::Vector{<:Real}; λ=1.0 )
    
    NS = length(x)
    sm(dp) = max(λ*dp,0.0)
    normalize!(x,1)

    T = [ sm(p[j]-p[i]) for i=1:NS, j=1:NS ] 

    x_dot = (T'*diagm(x)-diagm(x)*T)*ones(NS) 

    x_dot
end

function unbounded_smith_S(x::Vector{<:Real},p::Vector{<:Real}; λ=1.0, τ=1.0 )
    sum(x.*sum(max.(0,p.-p').^2,dims=1))
end


"""
`bnn(x::Vector{<:Real},p::Vector{<:Real}; λ=1.0, τ=1.0 ) -> Vector{<:Real}`

Brown-von Neumann-Nash (BNN) dynamics. Takes parameters `x` (a vector with the current state of the population
and `p` the currently offered payoff vector.

# Examples
```julia
(TODO)
julia> bnn(ones(ns),zeros(ns); λ=7.0, τ=3.0 )
...
```
"""
function bnn(x::Vector{<:Real},p::Vector{<:Real}; λ=1.0, τ=10000.0 )
    NS = length(x)
    normalize!(x,1)

    p_hat = p .- x'*p
    
    T = min.(λ.*max.(p_hat,0),τ)  
    
    x_dot  = T*sum(x)-x*sum(T)

    x_dot
end

#TODO fix the storage functions to respect the maximum switching τ
function bnn_S(x::Vector{<:Real},p::Vector{<:Real}; λ=1.0, τ=1.0 )
    sum(  min.(λ.*max.(p.-p'*x,0),τ).^2)/2
end





function hybrid_rule_maker(weights; learning_rules=[bnn, unbounded_smith])
    @assert length(weights)==length(learning_rules)
    @assert all(weights.>=0)
    @assert sum(weights)>0

    function hybrid_rule(x::Vector{<:Real},p::Vector{<:Real})
        sum( weights[i]*learning_rules[i](x,p) for (i,j) in enumerate(weights) )
    end
end


function hybrid_storage_fun_maker(weights; storage_funs=[bnn_S, unbounded_smith_S])
    @assert length(weights)==length(storage_funs)
    @assert all(weights.>=0)
    @assert sum(weights)>0


    function hybrid_rule_storage(x::Vector{<:Real},p::Vector{<:Real})
        sum( weights[i]*storage_funs[i](x,p) for (i,j) in enumerate(weights))
    end
end




##### immitation or replicator?