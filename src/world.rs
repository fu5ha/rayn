use crate::hitable::HitableStore;
use crate::material::MaterialStore;

pub struct World<S> {
    pub hitables: HitableStore<S>,
    pub materials: MaterialStore<S>,
}
