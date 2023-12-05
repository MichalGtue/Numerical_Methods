def cubic_spline(z,c):
    '''Generates a sevral cubic equations using spline interpolation'''
    Matrix = np.zeros((4*(len(z)-1), 4*(len(z)-1)))
    solution_vector_spline = np.zeros(4*len(z)-1)
    solution_vector_spline[0] = c[0]
    solution_vector_spline[2*len(c)-3] = c[-1]
    for i in range(len(c)-2): ## the first and last point are manually added
        solution_vector_spline[2*i + 1] = y[i+1] ## Filling in the y values
        solution_vector_spline[2*i + 2] = y[i+1]
    Matrix[0,3] = z[0]
    return solution_vector_spline ,  Matrix


