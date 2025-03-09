using StaticArrays
using LinearAlgebra
using DataStructures

# We define the Trajectory struct

struct Trajectory
    points::Vector{SVector{2, Float64}}     # The points of the trajectory where the importance function stricly increases
    max::Float64                    # The maximum value of the importance function on the trajectory
    id::Float64                     # The id of the trajectory
    n_calls::Int                # The number of calls to the gradient of the potential during the simulation
end

# We define the isless operator to create a strict order between Trajectories

Base.isless(x::Trajectory, y::Trajectory) = isless((x.max, x.id), (y.max, y.id))
Base.:(==)(a::Trajectory, b::Trajectory) = a.id == b.id


# The function transition computes the next position of the Markov chain following the discrete SDE

function transition(
    x::SVector{2, Float64}, 
    beta::Float64, 
    dt::Float64, 
    gradV_x::SVector{2, Float64}
)
    return x - dt * gradV_x + sqrt(2 * dt / beta) * SVector{2, Float64}(randn(2))
end

# The function simulate generate a random trajectory following the discrete SDE associated to the parameters in input

function simulate(
    x0::SVector{2, Float64}, 
    beta::Float64, 
    r_A::Float64, 
    r_B::Float64, 
    dt::Float64, 
    importance_func::Function, 
    gradV_func::Function, 
    id::Float64
)
    x = x0
    max_xi = importance_func(x0)
    chain = SVector{2, Float64}[x]
    n_calls = 0
    
    while r_A < norm(x) <= r_B
        # Compute the gradient of the potential at the current point
        gradV_x = gradV_func(x)

        # Update the number of calls to the gradient
        n_calls += 1

        # Compute the next point of the trajectory
        x = transition(x, beta, dt, gradV_x)

        # Store and update the maximum importance value only if it stricly increases
        if importance_func(x) > max_xi
            max_xi = importance_func(x)
            chain = push!(chain, x)
        end
    end
    
    return Trajectory(collect(chain), max_xi, id, n_calls)
end

# The function get_branching_points returns the first point of the trajectory 
# with importance stricly greater than level using log search

function get_branching_points(trajectory::Trajectory, level ::Float64, importance_func::Function)
    s::Int = 1
    t::Int = length(trajectory.points)
    while s < t
        if importance_func(trajectory.points[div(s + t, 2)]) < level
            s = div(s + t, 2) + 1
        else
            t = div(s + t, 2)
        end

    end
    return trajectory.points[t]
end

# The function calculate_probability computes the AMS estimator of the probability of the event {X_T in B}
# (we return in addition the number of itterations q and the number of calls to the force gradV)

function calculate_probability(
    n_trajectories::Int, 
    x0::SVector{2, Float64}, 
    beta::Float64, 
    r_A::Float64, 
    r_B::Float64, 
    dt::Float64, 
    k::Int, 
    Z_max::Float64,
    importance_func::Function,
    gradV_func::Function
)
    # Initialize the trajectories and weights
    # Note that we use an AVLTree to store the trajectories to maintain an order structure on the trajectories
    trajectories = AVLTree{Trajectory}()
    weights = Vector{Int}(undef, 0)
    id = 0.0

    # We simulate n_trajectories
    for i in 1:n_trajectories
        push!(trajectories, simulate(x0, beta, r_A, r_B, dt, importance_func, gradV_func, id))
        id += 1.0
    end

    # Initialize simulation variables
    Z::Float64 = 0.0
    q::Int = 0
    n_calls::Int = 0
    
    # Get the kth importance value
    Z = trajectories[k].max
    
    # In case every trajectory has importance less than Z we set Z to infinity to stop the algorithm
    if Z == trajectories[n_trajectories].max
        Z = Z_max + 1.0
    end
    
    # We start the simulation
    while Z <= Z_max
        i::Int = k
        
        # Compute the number of trajectories with importance less than Z
        while i+1 <= n_trajectories && trajectories[i+1].max <= Z
            i += 1
        end
        
        # Push the number of trajectories with importance less than Z into the weights vector
        push!(weights, i)
        
        # We simulate i new trajectories using the branching method
        for j in 1:i
            random_index::Int = rand(i+1:n_trajectories)
            branching_point = get_branching_points(trajectories[random_index], Z, importance_func)
            n_calls += trajectories[1].n_calls
            delete!(trajectories, trajectories[1])
            
            push!(trajectories, simulate(branching_point, beta, r_A, r_B, dt, importance_func, gradV_func, id))
            id += 1.0
        end
        
        # Update Z
        Z = trajectories[k].max
        
        # In case every trajectory has importance less than Z we set Z to infinity to stop the algorithm
        if Z == trajectories[n_trajectories].max
            Z = Z_max + 1.0
        end
        
        # Update q
        q += 1
    end
    
    # Compute the AMS estimator of the probability
    proba::Float64 = 0.0
    
    # If q is 0 we compute the probability directly as if it was MC
    if q == 0
        for i in 1:n_trajectories
            if trajectories[i].max >= Z_max
                proba += 1.0 / n_trajectories
                n_calls += trajectories[i].n_calls
            end
        end
    else
    # If q is not 0 we compute the probability using the weights
        G::Float64 = 1.0 / n_trajectories
        # We compute the weight of the final trajectories using the number of trajectories deleted at each step
        for i in 1:q
            G *= 1.0 - weights[i] / n_trajectories
        end
        
        for i in 1:n_trajectories
            if trajectories[i].max >= r_B
                proba += G
                n_calls += trajectories[i].n_calls
            end
        end
    end
    
    return proba, q, n_calls
end