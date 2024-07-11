using Measures



"""
`mutable struct Game`

F: calculates the current payoff, a function (sd::SystemDescription,state::Vector{Vector{Real}},t::Real) -> Vector{Real}
G: payoff dynamics, a function (sd::SystemDescription,state::Vector{Vector{Real}},t::Real) -> Vector{Real}
""" 
mutable struct Game
    ns::Int              #number of strategies
    F                    #payoff function (sd::SystemDescription,state::Vector{Vector{Real}},t::Real)
end



"""
`mutable struct PDM`

F: calculates the current payoff, a function (sd::SystemDescription,state::Vector{Vector{Real}},t::Real) -> Vector{Real}
G: payoff dynamics, a function (sd::SystemDescription,state::Vector{Vector{Real}},t::Real) -> Vector{Real}
""" 
mutable struct PDM
    ns::Int              #number of strategies
    nq::Int              #number of states for the PDM
    q_dot                #pdm dynamics function (sd::SystemDescription,pop_state::Vector{Vector{Real}},pdm_state::Vector{Vector{Real}},t::Real)
    F                    #payoff function (sd::SystemDescription,pop_state::Vector{Vector{Real}},pdm_state::Vector{Vector{Real}},t::Real)
end


"""
`simplex_plot!(s;ptype=scatter!)`

Generate a plot of the strategies evolving on the simplex, games must have 3 strategies.
""" 
function simplex_plot!(s;ptype=scatter!, plot_dst=nothing)

    if s isa Vector
        for sol in s
            simplex_plot_aux(sol; ptype=ptype, plot_dst=plot_dst)
        end
    else
        simplex_plot_aux(s; ptype=ptype, plot_dst=plot_dst)
    end
    if isnothing(plot_dst)
        plot!([0; 1; -1; 0],[1; -1/2 ;-1/2; 1], ticks=false, label=false)
    else
        plot!(plot_dst, [0; 1; -1; 0],[1; -1/2 ;-1/2; 1], ticks=false, label=false)
    end
end

"""
see simplex_plot!
""" 
function simplex_plot_aux(s; ptype=scatter!, plot_dst=nothing, plot_options...)
    t = s.t
    x = s.u


    if length(x[1])>3
        x = (v->v[1:3]).(x)
    end

    #if the list is too long downsample it rate to not use too much memory on pgfplotsx
    ds = 1
    if length(x)>3000
        global ds = round(Int,length(t)/3000)
    end
    t = t[1:ds:end]
    x = x[1:ds:end]


    x = map(v->[0 1 -1;1 -1/2 -1/2]*v,x)
    x = hcat(x...)
    

    if isnothing(plot_dst)
        ptype(x[1,:],x[2,:],label=false,lw=2.5; plot_options...)
        scatter!([x[1,end]],[x[2,end]],ms=7,shape=:circle,c=:black,label=false,showaxis = false)
    else
        ptype(plot_dst,x[1,:],x[2,:],label=false,lw=2.5; plot_options...)
        scatter!(plot_dst,[x[1,end]],[x[2,end]],ms=7,shape=:circle,c=:black,label=false,showaxis = false)
    end
end

"""
`simplex_quiver_plot!(s,f;ptype=scatter!)`

Generate a plot of the strategies evolving on the simplex, games must have 3 strategies.
Also prints the 
""" 
function simplex_quiver_plot!(s,F;ptype=scatter!,T::Int=20,linescolor=:Spectral_3)

    
    dt = 1//T
    X = [ [i;j;1-i-j] for i in range(0,1,step=dt) for j in range(0,1-i,step=dt)]
    Y = F.(X)
    FX = hcat(map(v->[0 1 -1;1 -1/2 -1/2]*v, X)...)
    FY = hcat(map(v->1.5*dt.*normalize([0 1 -1;1 -1/2 -1/2]*v,1), Y)...)
    #FZ = hcat(map(v->norm(v,2), Y)...)
    FZ = hcat(map(v->norm([0 1 -1;1 -1/2 -1/2]*v,1), Y)...)
    
    plot!([0; 1; -1; 0; 1].*1.02,[1; -1/2 ;-1/2; 1; -1/2].*1.02, ticks=false, label=false, c=:black, lw=3.0)
   
    #previous palette was :vik
    #should we use distinguishable_colors()?
    my_a = arrow(:closed, :head, 4, 6)
    if backend_name() == :pgfplotsx
        quiver!(FX[1,:], FX[2,:],quiver=(FY[1,:],FY[2,:]),
                #line_z=kron(FZ',ones(4)), 
                colorbar_title=L"\|\Phi p(t)\|_1", rightmargin=10mm, 
                colorbar_titlefontsize=25, arrow=my_a,
                c=:amp, lw=1, label=false)
    else
        quiver!(FX[1,:], FX[2,:],quiver=(FY[1,:],FY[2,:]),
                line_z=kron(FZ',ones(4)), 
                colorbar_title=L"\|\Phi p(t)\|_1", rightmargin=10mm, 
                colorbar_titlefontsize=25, arrow=my_a,
                c=:amp, lw=1, label=false)
    end



    if s isa Vector
        for (i, sol) in enumerate(s)
            simplex_plot_aux(sol; ptype=ptype, color=linescolor[1+(i-1)%length(linescolor)])
        end
    else
        simplex_plot_aux(s; ptype=ptype, color=linescolor)
    end

    plot!(ticks=false, grid=false)
     
end


function simplex_sols_plot!(s;ptype=scatter!,T::Int=20,linescolor=:Spectral_3, linestyles=[:dots] )

    
    plot!([0; 1; -1; 0; 1].*1.02,[1; -1/2 ;-1/2; 1; -1/2].*1.02, ticks=false, label=false, c=:black, lw=2.0)
    if s isa Vector
        for (i, sol) in enumerate(s)
            simplex_plot_aux(sol; ptype=ptype, color=linescolor[1+(i-1)%length(linescolor)], linestyle=linestyles[1+(i-1)%length(linestyles)])
        end
    else
        simplex_plot_aux(s; ptype=ptype, color=linescolor)
    end

    plot!(ticks=false, grid=false)
     
end

function rotation_mat2D(degrees)
    d = degrees*pi/180
    [cos(d) sin(d); -sin(d) cos(d)]
end

