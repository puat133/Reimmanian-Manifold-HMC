using LinearAlgebra
using Plots
using ReverseDiff: HessianTape, HessianConfig, hessian, hessian!, JacobianTape, JacobianConfig, jacobian, jacobian!, compile, CompiledTape
using ReverseDiff

"""
Some struct to make things tidy
"""
struct Target
    neglog_density::Function#the neglog_density function
    d::Integer#Dimension of the problem
    hessian_tape::Union{HessianTape,CompiledTape}
    hessian_result::Array{<:Real,2}
    u::AbstractArray{<:Real,1}#normalization parameter for modifiedCholesky
end


"""
The Soft absolute value function, with scalar inputs
"""
function sabs(x::Real,u::Real)
    log2 = log(2)
    temp = (x*log2/u)
    ret = log(exp(temp)+exp(-temp))*u/log2
    return ret
end

"""
Implementation of Modified Cholesky Decomposition routine
taken from :
`Modified Cholesky Riemann Manifold Hamiltonian Monte Carlo:
exploiting sparsity for fast sampling of high-dimensional targets`
    by Tore Selland Kleppe. DOI 10.1007/s11222-017-9763-5
"""
function modifiedCholesky(A::Symmetric{Float64,Array{Float64,2}},u::AbstractArray{<:Real,1},K::Integer=1)
    size_A = size(A)
    d = size_A[1]
    Id = Matrix{Float64}(I,size_A)
    L̃ = Id
    D = diag(A)
    for j in 1:size_A[1]
        if j>1
            L̃[j,1:j-1] = L̃[j,1:j-1]./D[1:j-1]

            if j<d
                L̃[j+1:d,j] = A[j+1:d,j]
                L̃[j+1:d,j] = L̃[j+1:d,j] - L̃[j+1:d,1:j-1]*L̃[j,1:j-1]
            end
        else
            if j<d
                L̃[j+1:d,j] = A[j+1:d,j]
            end
        end
        if j>K
            D[j] = sabs(D[j],u[j])
        end
        if j<d
            D[j+1:d] = D[j+1:d] - L̃[j+1:d,j].^2/D[j]
        end
    end
    L = LowerTriangular(L̃*diagm(D))
    return L
end

"""
the metric matrix function, evaluated for a specific negative log target probability distribution
"""
function metric(x::AbstractArray{<:Real,1},target::Target)
    H = Symmetric(hessian!(target.hessian_result,target.hessian_tape,x))
    L = modifiedCholesky(H,target.u)
    return L
end

"""
Hamiltonian of a Reimannian Manifold HMC, see eq 1 of
`Modified Cholesky Riemann Manifold Hamiltonian Monte Carlo:
exploiting sparsity for fast sampling of high-dimensional targets`
    by Tore Selland Kleppe. DOI 10.1007/s11222-017-9763-5
"""
# function hamiltonian(x::AbstractArray{<:Real,1},p::AbstractArray{<:Real,1},u::AbstractArray{<:Real,1},
#     neglog_target_pi,hessian_tape::HessianTape,hessian_result::Array{<:Real,2})
function hamiltonian(x::AbstractArray{<:Real,1},p::AbstractArray{<:Real,1},target::Target)

    L = metric(x,target)
    # L = L̃*diagm(D)
    p̂ = L\p
    ham = target.neglog_density(x) + 0.5p̂'*p̂ + logdet(L)
    return ham
end





struct Hamiltonian
    target::Target
    fun::Function
    jacobian!::Function
    jacobian_tape::Union{JacobianTape,CompiledTape}
    jacobian_result::Array{<:Real,1}
    Hamiltonian(target) = begin
        fun(x,p) = hamiltonian(x,p,target)
        x,p = randn(target.d),randn(target.d)
        jacobian_tape = JacobianTape(fun,(x,p))
        jacobian_result = similar(x)
    end
end

"""
Explicit Leap Frog algorithm: using 10.1103/PhysRevE.94.043303
"""
function generalized_leap_frog(ϵ::Real,ℓ::Integer,ω::Real
                            ,x::AbstractArray{<:Real,1},p::AbstractArray{<:Real,1},
                            target::Target)
    c = cos(2ω*ϵ)
    s = sin(2ω*ϵ)

    x̃ = copy(x)
    p̃ = copy(p)

    for i in 1:ℓ
        #2
        jacobian!(target.jacobian_result,jacobian_tape,(x,p̃))
        p -= 0.5ϵ*(target.jacobian_result)

        #3
        L = metric(x,u,target.hessian_tape,target.hessian_result)
        x̃ += 0.5ϵ*(L'\L\p̃) #this is more efficient

        #4
        jacobian!(target.jacobian_result,target.jacobian_tape,(x̃,p))
        p̃ -= 0.5ϵ*(target.jacobian_result)

        #5
        L = metric(x̃,u,target.hessian_tape,target.hessian_result)
        x += 0.5ϵ*(L'\L\p) #this is more efficient

        #6 no need to compute c = cos(2ωϵ); s = sin(2ωϵ) every timestep
        #7
        x = (x+x̃ + c*(x-x̃) + s*(p-p̃))/2

        #8
        p = (p+p̃ - s*(x-x̃) + c*(p-p̃))/2

        #9
        x̃ = (x+x̃ - c*(x-x̃) - s*(p-p̃))/2

        #10
        p̃ = (p+p̃ + s*(x-x̃) - c*(p-p̃))/2

        #11
        jacobian!(target.jacobian_result,target.jacobian_tape,(x̃,p))
        p̃ -= 0.5ϵ*(target.jacobian_result)

        #12
        L = metric(x̃,u,target.hessian_tape,target.hessian_result)
        x += 0.5ϵ*(L'\L\p) #this is more efficient

        #13
        jacobian!(target.jacobian_result,target.jacobian_tape,(x,p̃))
        p -= 0.5ϵ*(target.jacobian_result)

        #14
        L = metric(x,u,target.hessian_tape,target.hessian_result)
        x̃ += 0.5ϵ*(L'\L\p̃) #this is more efficient
    end

    return x,p
end

"""
Example of target density
"""
function funnel_neglog(x::AbstractArray{<:Real,1})
    nLP = (x[1]^2/(2exp(x[2])) + x[2]/2 + x[2]^2/18)
    # nLP = (x^2/(2exp(y)) + y/2 + y^2/18)
    return nLP
end








n = 2
x = randn(n)
p = randn(n)
w = randn(n)
# hessConfig = HessianConfig(x)
hessian_result = similar(x,n,n)

ϵ = 0.5n^(-0.25)
ℓ = 1.5/ϵ

target_pi = funnel_neglog
hess_tape = HessianTape(target_pi,x)
compiled_hess_tape = compile(hess_tape)
hessian!(hessian_result,compiled_hess_tape,x)
u = eigmin(hessian_result)ones(n)
target = Target(funnel_neglog,n,compiled_hess_tape,hessian_result,abs.(u))
target_hamiltonian(x::AbstractArray{<:Real,1},p::AbstractArray{<:Real,1}) = hamiltonian(x::AbstractArray{<:Real,1},p::AbstractArray{<:Real,1},target::Target)

jacobian_tape = JacobianTape(target_hamiltonian,(x,p))
compiled_jac_tape = compile(jacobian_tape)
jacobian_result = similar(x,n,n)
jacobian!(jacobian_result,compiled_jac_tape,x,p)

L = modifiedCholesky(Symmetric(hessian_result),u,1)
L'*L




# #test function
# function negLogPosterior(x)
#     return sqrt(sum(x.^2))
# end

# """
# Julia does not have a meshgrid function?
# """
# function meshgrid(x, y)
#     return (repeat(x',(length(y)),1), repeat(y,1,(length(x))))
# end




#

#
# u = [1:n...]/1.
#
# L̃,D = modifiedCholesky(Symmetric(H),u,1)
#
# x = [-5:0.1:30...]
# y=[-5:0.1:5...]
# X,Y = meshgrid(x,y)
# Z = @. exp(-(X^2/(2exp(Y)) + Y/2 + Y^2/18))
# heatmap(x,y,Z,fill=(true,:greys_r))
