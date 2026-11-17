#include "py_lasyten_nanobind.h"

namespace lasyten {

void bind_symmetries_exceptions(nb::module_& m);
void bind_symmetries_styles(nb::module_& m);
void bind_sector(nb::module_& m);
void bind_base_symmetry(nb::module_& m);
void bind_symmetry_factor(nb::module_& m);
void bind_group(nb::module_& m);
void bind_abelian_group(nb::module_& m);
void bind_no_symmetry(nb::module_& m);
void bind_u1(nb::module_& m);
void bind_zn(nb::module_& m);
void bind_su2(nb::module_& m);
void bind_sun(nb::module_& m);
void bind_fermion_number(nb::module_& m);
void bind_fermion_parity(nb::module_& m);
void bind_zn_anyon_category(nb::module_& m);
void bind_zn_anyon_category2(nb::module_& m);
void bind_quantum_double_zn_anyon_category(nb::module_& m);
void bind_toric_code_category(nb::module_& m);
void bind_fibonacci_anyon_category(nb::module_& m);
void bind_ising_anyon_category(nb::module_& m);
void bind_su2_k_anyon_category(nb::module_& m);
void bind_su3_3_anyon_category(nb::module_& m);
void bind_symmetry(nb::module_& m);
void bind_trees(nb::module_& m);
void bind_spaces(nb::module_& m);

void
bind_symmetries(nb::module_& m)
{
    bind_symmetries_exceptions(m);
    bind_symmetries_styles(m);
    bind_sector(m);
    bind_base_symmetry(m);
    bind_symmetry_factor(m);
    bind_group(m);
    bind_abelian_group(m);
    bind_no_symmetry(m);
    bind_u1(m);
    bind_zn(m);
    bind_su2(m);
    bind_sun(m);
    bind_fermion_number(m);
    bind_fermion_parity(m);
    bind_zn_anyon_category(m);
    bind_zn_anyon_category2(m);
    bind_quantum_double_zn_anyon_category(m);
    bind_toric_code_category(m);
    bind_fibonacci_anyon_category(m);
    bind_ising_anyon_category(m);
    bind_su2_k_anyon_category(m);
    bind_su3_3_anyon_category(m);
    bind_symmetry(m);
    bind_trees(m);
    bind_spaces(m);
}

} // namespace lasyten
