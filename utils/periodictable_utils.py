import periodictable

def get_symbol_by_atomic_number(atomic_number):
    return periodictable.elements[atomic_number].symbol

def get_name_by_atomic_number(atomic_number):
    return periodictable.elements[atomic_number].name