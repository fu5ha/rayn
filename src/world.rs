use crate::camera::CameraStore;
use crate::hitable::HitableStore;
use crate::light::Light;
use crate::material::MaterialStore;

pub struct World {
    pub hitables: HitableStore,
    pub lights: Vec<Box<dyn Light>>,
    pub materials: MaterialStore,
    pub cameras: CameraStore,
}
