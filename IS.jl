using SpecialFunctions
using StaticArrays
using LinearAlgebra

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
    x0::SVector{2, Float64}, 
    beta::Float64, 
    r_A::Float64, 
    r_B::Float64, 
    dt::Float64, 
    gradV_func::Function, 
    gradb_func::Function
)

    x = x0
    arg::Float64 = 0.0
    gamma::Float64 = 2.0 / beta
    n_calls::Int = 0
    
    while r_A < norm(x) <= r_B
        # Compute the gradient of the biased potential, make only one call to gradb
        gradb_x = gradb_func(x, beta)
        gradV_x = gradV_func(x)
        n_calls += 1
        # Simulate the next step
        y = transition_IS(x, beta, dt, gradV_x, gradb_x)

        # Compute probability arg of the ratio between original and importance transitions
        temp = ((y - x) + gradV_x*dt + gradb_x/2.0*dt)/gamma
        arg += temp[1] * gradb_x[1] + temp[2] * gradb_x[2]

        # Update the chain
        x = y
    end
    
    # Return the final point and the likelihood ratio
    return x, exp(arg), n_calls

end

# The function important_sampling computes the importance sampling estimator of the probability of the event {X_T in B}

function important_sampling(
    x0::SVector{2, Float64}, 
    beta::Float64, r_A::Float64, 
    r_B::Float64, 
    dt::Float64, 
    gradV_func::Function, 
    gradb::Function, 
    n_max::Int
)

    comp::Float64 = 0.0
    n_calls::Int = 0

    for j in 1:n_max
        x,likelihood, n_calls_temp = simulate(x0, beta, r_A, r_B, dt, gradV_func, gradb)
        n_calls += n_calls_temp
        if norm(x) >= r_B
            comp += likelihood
        end
    end
    return comp/n_max, n_calls/n_max
end


# Define the gradient of the biased potential for IS

function grad_comitor(
    x::SVector{2, Float64}, 
    beta::Float64, 
    r_A::Float64=1.
)

    r=norm(x)
    
    # Calculate radial component of gradient
    grad_r = - 4 * exp(beta * r^2 / 2)/ r / beta / (expinti(beta * r^2 / 2 ) - expinti(beta * r_A^2 / 2))
    
    # Return gradient vector
    grad_return::SVector{2, Float64} = SVector{2, Float64}([grad_r * x[1] / r, grad_r * x[2] / r])
    return grad_return
end