use crate::material::MaterialHandle;
use crate::math::{Mat3, OrthonormalBasis, Vec3};
use crate::ray::Ray;

#[derive(Clone, Copy)]
pub struct Intersection {
    pub t: f32,
    pub point: Vec3,
    pub offset_by: f32,
    pub normal: Vec3,
    pub basis: Option<Mat3>,
    pub material: MaterialHandle,
    // pub eta: f32,
}

impl Intersection {
    pub fn new(
        t: f32,
        point: Vec3,
        offset_by: f32,
        normal: Vec3,
        material: MaterialHandle,
    ) -> Self {
        Intersection {
            t,
            point,
            offset_by,
            normal,
            basis: None,
            material,
        }
    }

    pub fn basis(&mut self) -> Mat3 {
        if let Some(basis) = self.basis {
            basis
        } else {
            let basis = self.normal.get_orthonormal_basis();
            self.basis = Some(basis);
            basis
        }
    }

    pub fn create_ray(&self, dir: Vec3) -> Ray {
        Ray::new(
            self.point + self.normal * self.normal.dot(dir).signum() * self.offset_by,
            dir,
        )
    }
}

pub trait Hitable: Send + Sync {
    fn hit(&self, ray: &Ray, time: f32, t_range: ::std::ops::Range<f32>) -> Option<Intersection>;
}

pub struct HitableStore(Vec<Box<dyn Hitable>>);

impl HitableStore {
    pub fn new() -> Self {
        HitableStore(Vec::new())
    }

    pub fn push(&mut self, hitable: Box<dyn Hitable>) {
        self.0.push(hitable)
    }
}

impl ::std::ops::Deref for HitableStore {
    type Target = Vec<Box<dyn Hitable>>;

    fn deref(&self) -> &Vec<Box<dyn Hitable>> {
        &self.0
    }
}

impl Hitable for HitableStore {
    fn hit(&self, ray: &Ray, time: f32, t_range: ::std::ops::Range<f32>) -> Option<Intersection> {
        self.iter()
            .fold((None, t_range.end), |acc, hitable| {
                let mut closest = acc.1;
                let hr = hitable.hit(ray, time, t_range.start..closest);
                if let Some(Intersection { t, .. }) = hr {
                    closest = t;
                }
                let hr = if hr.is_some() { hr } else { acc.0 };
                (hr, closest)
            })
            .0
    }
}
