# # MATH50003 (2024–25)
# # Lab 9: VI.1 General Orthogonal Polynomials and VI.2 Classical Orthogonal Polynomials


# This lab investigates the numerical construction of orthogonal polynomials, and
# the practical usage of classical orthogonal polynomials. There is a special emphasis
# on Chebyshev polynomials which are directly related to Cosine and Sine expansions.

# 
# **Learning Outcomes**
#
# Mathematical knowledge:

# 1. Gram–Schmidt for continuous functions

# Coding knowledge:

# 1. 

# We use the following packages:

using QuadGK, Plots, LinearAlgebra, Test


# ## VI.1 General Orthogonal Polynomials

# Orthogonal polynomials are graded polynomials which have the form
# $$
# p_n(x) = k_n x^n + k_n^{(1)} x^{n-1} + ⋯ + k_n^{(n-1)} x + k_n^{(n)}
# $$
# We can store the (currently unknown) coefficients of the orthogonal polynomials as an upper-triangular matrix:
# $$
# R_n = \begin{bmatrix} k_0 & k_1^{(1)} & k_2^{(2)} & ⋯ k_n^{(n)} \\
#               & k_1 & k_2^{(1)} & ⋯ & k_n^{(n-1)} \\
#                &  & ⋱ & ⋱ & ⋮ \\
#                 & & & k_{n-1} & k_n^{(1)} \\
#                   &&&& k_n
# \end{bmatrix}
# $$
# This can be written as
# $$
# [p_0| …| p_n] = [1| x| …| x^n] R_n
# $$

# We can build monic orthogonal polynomials using a continuous version of Gram–Schmidt:
# $$
#  π_n(x) = x^n - ∑_{k=0}^{n-1} {⟨x^n,π_k ⟩ \over \|π_k\|^2} π_k(x).
# $$
# To deduce $R$ from this process we proceed as follows, assuming the inner product is
# $$
# ⟨f,g⟩ := ∫_0^1 f(x) g(x) w(x) {\rm d}x
# $$
# which we approximate with `quadgk`:

function opgramschmidt(w, n)
    R = UpperTriangular(zeros(n,n)) # Connection matrix with monomials
    for j = 1:n
        R[j,j] = 1 # k_j = 1
        for k = 1:j-1
            πₖ = x -> R[1:k,k]'*[x^ℓ for ℓ=0:k-1] # the previously computed OP
            ip = quadgk(x -> x^(j-1) * πₖ(x) * w(x), 0, 1)[1] # ⟨x^n,π_k⟩ 
            nrm = quadgk(x -> πₖ(x)^2 * w(x), 0, 1)[1] # ||π_k||^2. A better version would store this as its repeating the computation for each j
            R[1:k,j] -= ip/nrm * R[1:k,k] # R[1:k,k] gives us the monomial expansion of πₖ
        end
    end
    R
end

# For the special case of $w(x) = 1$ we get:

opgramschmidt(x -> 1, 5)

# That is, we have computed
# $$
# π_0(x) = 1, π_1(x) = x-1/2, π_2(x) = x^2 - x + 1/6, π_3(x) = x^3 - 3x^2/2 + 3x/5 - 1/20
# $$
# which example match the explicit computation from the notes.

# ----

# **Problem 1(a)** Modify `opgramschmidt` to take in the support of the inner product $(a,b)$ and
# not recompute $\|π_k\|^2$ multiple times, and return a tuple containing $R$ and a vector containing $\|π_0\|^2,…,\|π_{n-1}\|^2$.

function opgramschmidt(w, n, a, b)
    ## TODO: Modify the above code to support general weights and not recompute ||π_k||^2
    ## SOLUTION
    R = UpperTriangular(zeros(n,n)) # Connection matrix with monomials
    nrms = zeros(n) # vector of inner products
    for j = 1:n
        R[j,j] = 1 # k_j = 1
        for k = 1:j-1
            πₖ = x -> dot(R[1:k,k],[x^ℓ for ℓ=0:k-1]) # the previously computed OP
            ip = quadgk(x -> x^(j-1) * πₖ(x) * w(x), a, b)[1] # ⟨x^n,π_k⟩ 
            R[1:k,j] -= ip/nrms[k] * R[1:k,k] # R[1:k,k] gives us the monomial expansion of πₖ
        end
        πⱼ = x -> dot(R[1:j,j],[x^ℓ for ℓ=0:j-1]) # the previously computed OP
        nrms[j] =  quadgk(x -> πⱼ(x)^2 * w(x), a, b)[1]
    end
    R,nrms
    ## END
end

R,nrms = opgramschmidt(x -> 1, 3, 0, 1)
@test R ≈ [1 -1/2 1/6;
           0  1   -1;
           0  0    1]
@test nrms ≈ [1,1/12,1/180]

# **Problem 1(b)** Use the new `opgramschmidt` to compute the monic OPs for $\sqrt{1-x^2}$ and $1-x$ on $[-1,1]$
# Do these match the computation from the problem sheet?

## TODO: employ the new opgramschmidt to the two examples from the problem sheet. 
## SOLUTION
opgramschmidt(x -> sqrt(1-x^2), 5, -1, 1)[1]
## Yes it matches 1, x, x^2 - 1/4, x^3 - x/2
opgramschmidt(x -> 1-x, 5, -1, 1)[1]
## Yes it matches 1, x+1/3, x^2 + 2x/5 - 1/5, x^3 + 3x^2/7 - 3x/7 - 3/35
## END

# **Problem 1(c)** By calling `opgramschmidt` implement `orthonormalgramschmidt`
# to return the corresponding orthonormal polynomials. For the two examples in the previous problem,
# does this match what you derived in the problem sheet?


# -----

# ### VI.1.1 Three-term recurrence

# ## VI.2 Classical Orthogonal Polynomials