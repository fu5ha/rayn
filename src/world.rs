use crate::hitable::HitableStore;
use crate::material::MaterialStore;
use crate::camera::CameraStore;

pub struct World<S> {
    pub hitables: HitableStore<S>,
    pub materials: MaterialStore<S>,
    pub cameras: CameraStore,
}
