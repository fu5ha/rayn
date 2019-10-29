use crate::camera::CameraStore;
use crate::hitable::HitableStore;
use crate::material::MaterialStore;

pub struct World {
    pub hitables: HitableStore,
    pub materials: MaterialStore,
    pub cameras: CameraStore,
}
