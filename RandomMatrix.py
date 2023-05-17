import numpy as np
from scipy.special import lambertw
from scipy import linalg

def generate_random_matrix(S, C, sigma, dB):
    '''
    Generate a random matrix with a constant diagonal dB, connectance C and variance sigma.
    
    Parameters
    ----------
    S : int
        Number of species.
    C : float  
        Connectance of the matrix.
    sigma : float   
        Variance of the matrix.
    dB : float
        - Diagonal element of the delayed Jacobian matrix.

    Returns    
    -------
    B : array_like
        Random matrix.
    '''
    
    B = np.zeros((S, S))
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            if np.random.rand() < C:
                B[i, j] = np.random.normal(0, sigma)
    np.fill_diagonal(B, -dB)
    return B

def generate_diagonal_matrix(S, d):
    '''
    Generate a diagonal matrix with a constant diagonal d.
    
    Parameters
    ----------
    S : int
        Number of species.
    d : float  
        Diagonal element.
    
    Returns
    -------
    A : array_like
        Diagonal matrix.
    '''
    
    A = np.zeros((S, S))
    np.fill_diagonal(A, d)
    return A

def eigenvalues_discrete_delay(A, B, tau):
    '''
    Compute the eigenvalues of the discrete delay equation.
    
    Parameters
    ----------
    A : array_like
        non-delayed Jacobian matrix.
    B : array_like
        delayed Jacobian matrix.
    tau : float
        delay.
        
    Returns
    -------
    eigenvalues : array_like
        eigenvalues of the discrete delay equation.
    '''
    
    # Compute the eigenvalues of the discrete delay matrix
    eigenvaluesB = linalg.eigvals(B)
    # Compute the eigenvalues of the non-delayed matrix (since it is diagonal, it is just the diagonal elements)
    eigenvaluesA = A[0,0]
    # Compute the eigenvalues of the discrete delay equation
    if tau == 0:
        eigenvalues = eigenvaluesA + eigenvaluesB
    elif tau > 0:
        eigenvalues = eigenvaluesA+lambertw(eigenvaluesB*tau*np.exp(-eigenvaluesA*tau))/tau
    return eigenvalues


def discrete_system(dA, dB, S, C, sigma, tau):
    '''
    Function to compute the eigenvalues of the discrete delay equation.
    
    Parameters
    ----------
    dA : float
        - Diagonal element of the non-delayed Jacobian matrix.
    dB : float
        - Diagonal element of the delayed Jacobian matrix.
    S : int
        Number of species.
    C : float
        Connectance of the matrix.
    sigma : float
        Variance of the matrix.
    tau : float
        Delay.
        
    Returns
    -------
    eigenvalues : array_like
        eigenvalues of the discrete delay equation.
    '''
    # Generate the non-delayed and delayed Jacobian matrices
    A = generate_diagonal_matrix(S, -dA)
    B = generate_random_matrix(S, C, sigma, dB)
    # Compute the eigenvalues of the discrete delay equation
    eigenvalues = eigenvalues_discrete_delay(A, B, tau)
    return eigenvalues


def eigenvalues_exponential_delay(A, B, tau):
    '''
    Compute the eigenvalues of the discrete delay equation.
    
    Parameters
    ----------
    A : array_like
        non-delayed Jacobian matrix.
    B : array_like
        delayed Jacobian matrix.
    tau : float
        average delay of the exponential distribution.
        
    Returns
    -------
    eigenvalues : array_like
        eigenvalues of the discrete delay equation.
    '''
    
    # Compute the eigenvalues of the discrete delay matrix
    eigenvaluesB = linalg.eigvals(B)
    # Compute the eigenvalues of the non-delayed matrix (since it is diagonal, it is just the diagonal elements)
    eigenvaluesA = A[0,0]
    # Compute the eigenvalues of the discrete delay equation
    
    mean_rate = 1 / tau
    eigenvalues_up   = -0.5*(-eigenvaluesA+mean_rate)+np.sqrt(0.25*(-eigenvaluesA+mean_rate)**2+mean_rate*(eigenvaluesB+eigenvaluesA))
    eigenvalues_down = -0.5*(-eigenvaluesA+mean_rate)-np.sqrt(0.25*(-eigenvaluesA+mean_rate)**2+mean_rate*(eigenvaluesB+eigenvaluesA))
    eigenvalues = np.concatenate((eigenvalues_up,eigenvalues_down))

    return eigenvalues


def exponential_system(dA, dB, S, C, sigma, tau):
    '''
    Function to compute the eigenvalues of the discrete delay equation.
    
    Parameters
    ----------
    dA : float
        - Diagonal element of the non-delayed Jacobian matrix.
    dB : float
        - Diagonal element of the delayed Jacobian matrix.
    S : int
        Number of species.
    C : float
        Connectance of the matrix.
    sigma : float
        Variance of the matrix.
    tau : float
        Average delay of the exponential distribution.
        
    Returns
    -------
    eigenvalues : array_like
        eigenvalues of the exponentially distributed delay equation.
    '''
    # Generate the non-delayed and delayed Jacobian matrices
    A = generate_diagonal_matrix(S, -dA)
    B = generate_random_matrix(S, C, sigma, dB)
    # Compute the eigenvalues of the discrete delay equation
    eigenvalues = eigenvalues_exponential_delay(A, B, tau)
    return eigenvalues