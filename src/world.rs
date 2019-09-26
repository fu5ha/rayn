use crate::hitable::HitableStore;
use crate::material::MaterialStore;
use crate::camera::Camera;

pub struct World<S> {
    pub hitables: HitableStore<S>,
    pub materials: MaterialStore<S>,
    pub camera: Box<dyn Camera>,
}
