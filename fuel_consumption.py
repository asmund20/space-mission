# Denne funksjonen beregner hvor mye drivstoff som forbrennes
def fuel_consumed(F, consumption, m, dv) -> float:
    return consumption*m*dv/F
