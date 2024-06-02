import numpy as np

def compact_complex(z, sign_digits=3):
    # This function prints a (complex)
    def format_number(n):
        if abs(n) < 1e-3 or abs(n) >= 1e3:  # Scientific notation for very small or large numbers
            return f"{n:.{sign_digits}g}"   # Use scientific notation with given significant digits
        else:
            return f"{n:.{sign_digits}g}"   # Use up to given significant digits

    # Convert integer to complex number with zero imaginary part if necessary
    if isinstance(z, int):
        z = complex(z, 0)
    
    real_part = int(z.real) if z.real.is_integer() else z.real
    imag_part = int(z.imag) if z.imag.is_integer() else z.imag

    real_str = f"{format_number(real_part)}" if real_part != 0 else ""
    
    if imag_part != 0:
        if real_part != 0:
            imag_sign = ' + ' if imag_part > 0 else ' - '
            imag_str = f"{imag_sign}{format_number(abs(imag_part))}j"
        else:
            imag_str = f"{format_number(imag_part)}j"
    else:
        imag_str = ""

    if real_str and imag_str:
        return f"{real_str}{imag_str}"
    elif real_str:
        return real_str
    elif imag_str:
        return imag_str
    else:
        return "0"



def isinteger(x):
    return np.equal(np.mod(x, 1), 0)



