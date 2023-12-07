# Steady-state assumption for the mass balance (Accumulation is approx. 0)
# Derivations can be found in the report, here only the final answers are given

print("Rearranged equations for the concentrations of PCBs in various lakes: \n\n Lake Superior: C_S = S_in / Q_SH \n\n Lake Michigan : C_M=S_(in,M)/Q_MH \n\nLake Huron: C_H=S_(in,H)/Q_HE +S_in/Q_HE +S_(in,M)/Q_HE \n\n Lake Erie: C_E=S_(in,E)/Q_EO +S_(in,H)/Q_EO +S_in/Q_EO +S_(in,M)/Q_EO \n\n Lake Ontario: C_O=S_(in,O)/Q_OO +S_(in,E)/Q_OO +S_(in,H)/Q_OO +S_in/Q_OO +S_(in,M)/Q_OO \n")

print("The mass balances for the lakes read as follows, \n\n Lake Superior: S_(in,S)=Q_SH⋅C_S \n\n Lake Michigan : S_(in,M)=Q_MH⋅C_M \n\n Lake Huron: S_(in,H)=-Q_SH⋅C_S-Q_MH⋅C_M+C_H⋅Q_HE \n\n Lake Erie:S_(in,E)=-Q_HE⋅C_H+Q_EO⋅C_E \n\n Lake Ontario: S_(in,O)=-Q_EO⋅C_E+Q_OO⋅C_O \n")