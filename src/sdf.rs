//! This module contains various functions for basic signed distance field primitives.
//! All of them are centered around the origin, so it is necessary to transform the point into
//! object local space of the primitive to use them.
use crate::math::{ Vec3, Vec2 };

pub fn sd_sphere(p: Vec3, radius: f32) -> f32 {
    p.magnitude() - radius
}

pub fn sd_box(p: Vec3, dims: Vec3) -> f32 {
    let d = p.map(|x| x.abs()) - dims;
    Vec3::partial_max(d, Vec3::broadcast(0.0)).magnitude()
        + d.y.max(d.z).max(d.z).min(0.0)
}

pub fn sd_torus(p: Vec3, radius: f32, thickness: f32) -> f32 {
    let q = Vec2::new(Vec2::new(p.x, p.z).magnitude() - thickness, p.y);
    q.magnitude() - radius
}

pub fn sd_cylinder_x(p: Vec3, radius: f32) -> f32 {
    Vec2::new(p.y, p.z).magnitude() - radius
}

pub fn sd_cylinder_y(p: Vec3, radius: f32) -> f32 {
    Vec2::new(p.x, p.z).magnitude() - radius
}

pub fn sd_cylinder_z(p: Vec3, radius: f32) -> f32 {
    Vec2::new(p.x, p.y).magnitude() - radius
}

const EPSILON: f32 = 0.0001;
pub fn estimate_normal<F: Fn(Vec3) -> f32>(p: Vec3, sdf: F) -> Vec3 {
    Vec3::new(
        sdf(Vec3::new(p.x + EPSILON, p.y, p.z)) - sdf(Vec3::new(p.x - EPSILON, p.y, p.z)),
        sdf(Vec3::new(p.x, p.y + EPSILON, p.z)) - sdf(Vec3::new(p.x, p.y - EPSILON, p.z)),
        sdf(Vec3::new(p.x, p.y, p.z + EPSILON)) - sdf(Vec3::new(p.x, p.y, p.z - EPSILON))
    ).normalized()
}