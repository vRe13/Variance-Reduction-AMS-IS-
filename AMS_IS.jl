using StaticArrays
using LinearAlgebra
using SpecialFunctions 
using DataStructures

# Define the Trajectory struct : 

struct Trajectory_IS
    points::Vector{SVector{2, Float64}}     # Points of the trajectory where the importance function stricly increases
    max::Float64                        # Maximum value of the importance function on the trajectory
    likelihood_arg::Vector{Float64}     # Likelihood argument of the trajectory
    id::Float64                    # Id of the trajectory
    n_calls::Int                # Number of calls to the gradient of the potential during the simulation
end

# We define the isless operator to create a strict order between Trajectories

Base.isless(x::Trajectory_IS, y::Trajectory_IS) = isless((x.max, x.id), (y.max, y.id))
Base.:(==)(a::Trajectory_IS, b::Trajectory_IS) = a.id == b.id


# The function transition computes the next position of the Markov chain following the discrete SDE

function transition_IS(
    x::SVector{2, Float64}, 
    beta::Float64, 
    dt::Float64, 
    gradV_x::SVector{2, Float64}, 
    gradb_x::SVector{2, Float64}
)
    return x - dt * (gradV_x + gradb_x) + sqrt(2 * dt / beta) * SVector{2, Float64}(randn(2))
end

# The function simulate generate a random trajectory following the discrete SDE associated to the parameters in input

function simulate(
    x0::SVector{2, Float64},  # x_0 the starting point that can be a branching point
    beta::Float64, 
    r_A::Float64, 
    r_B::Float64, 
    dt::Float64, 
    importance_func::Function, 
    gradV_func::Function, 
    gradb::Function, 
    id::Float64, 
    likelihood_arg_start::Float64=0.0       # Likelihood_arg of the branching point, 0.0 per default if we are not simulating from a branching point
)
    x = x0
    max_xi = importance_func(x0)
    chain = SVector{2, Float64}[x]
    likelihood_arg::Vector{Float64} = [likelihood_arg_start]
    likelihood_arg_temp::Float64 = likelihood_arg_start
    gamma::Float64 = 2.0 / beta
    n_calls = 0

    while r_A < norm(x) <= r_B
        # Compute the gradient of the potential and the biased potential, only one call to each during one step of the loop
        gradb_x = gradb(x, beta)
        gradV_x = gradV_func(x)

        # Count the number of calls to the gradient of the potential
        n_calls += 1 

        # Simulate the next step
        y = transition_IS(x, beta, dt, gradV_x, gradb_x)

        # Compute probability arg of the ratio between original and importance transitions and update the likelihood arg
        temp = ((y - x) + gradV_x*dt + gradb_x/2.0*dt)/gamma
        likelihood_arg_temp += temp[1] * gradb_x[1] + temp[2] * gradb_x[2]

        # In case the importance stricly increase update the chain and the likelihood arg
        if importance_func(y) > max_xi
            max_xi = importance_func(y)
            chain = push!(chain, y)
            push!(likelihood_arg, likelihood_arg_temp)
        end
        
        # Update the current point
        x = y
    end
    
    return Trajectory_IS(collect(chain), max_xi, likelihood_arg, id, n_calls)
end

# The function get_branching_points returns the first point of the trajectory 
# with importance stricly greater than level using log search

function get_branching_points(
    trajectory::Trajectory_IS, 
    level ::Float64, 
    importance_func::Function
)

    s::Int = 1
    t::Int = length(trajectory.points)
    while s < t
        if importance_func(trajectory.points[div(s + t, 2)]) < level
            s = div(s + t, 2) + 1
        else
            t = div(s + t, 2)
        end

    end
    return trajectory.points[t], trajectory.likelihood_arg[t]
end

# The function calculate_probability computes the AMS-IS estimator of the probability of the event {X_T in B}
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
    gradV_func::Function,
    gradb::Function
)
    # Initialize the trajectories and weights
    # Note that we use an AVLTree to store the trajectories to maintain an order structure on the trajectories
    trajectories = trajectories = AVLTree{Trajectory_IS}()
    weights = Vector{Int}(undef, 0)
    id = 0.0
    
    # We simulate n_trajectories
    for i in 1:n_trajectories
        push!(trajectories, simulate(x0, beta, r_A, r_B, dt, importance_func, gradV_func, gradb, id))
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
            n_calls += trajectories[1].n_calls
            delete!(trajectories, trajectories[1])
        end

        for j in 1:i
            random_index::Int = rand(1:n_trajectories - i)
            branching_point, branching_likelihood_arg = get_branching_points(trajectories[random_index], Z, importance_func)
            push!(trajectories, simulate(branching_point, beta, r_A, r_B, dt, importance_func, gradV_func, gradb, id, branching_likelihood_arg))
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
    
    # Compute the AMS-IS estimator of the probability
    proba::Float64 = 0.0
    
    # If q is 0 we compute the probability directly as if it was MC
    if q == 0
        for i in 1:n_trajectories
            if trajectories[i].max >= r_B
                proba += 1.0 / n_trajectories * exp(trajectories[i].likelihood_arg[end])
                n_calls += trajectories[i].n_calls
            end
        end
    else
    # If q is not 0 we compute the probability using the weights
        G::Float64 = 1.0 / n_trajectories
        # We compute the weight of the final trajectories using the number of trajectories deleted at each step
        for i in 1:q
            G *= 1.0  - weights[i] / n_trajectories
        end
        
        for i in 1:n_trajectories
            if trajectories[i].max >= r_B 
                proba += G * exp(trajectories[i].likelihood_arg[end])
                n_calls += trajectories[i].n_calls
            end
        end
    end
    
    return proba,q, n_calls
end
