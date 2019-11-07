use crate::material::{MaterialHandle, MaterialStore};
use crate::math::{OrthonormalBasis, Vec2u, Vec3, Wat3, Wec3};
use crate::ray::{Ray, WRay};

use wide::f32x4;

use bumpalo::collections::Vec as BumpVec;
use bumpalo::Bump;

#[derive(Clone, Copy)]
pub struct Intersection {
    pub ray: Ray,
    pub t: f32,
    pub point: Vec3,
    pub offset_by: f32,
    pub normal: Vec3,
}

impl Intersection {
    pub fn new(ray: Ray, t: f32, point: Vec3, offset_by: f32, normal: Vec3) -> Self {
        Intersection {
            ray,
            t,
            point,
            offset_by,
            normal,
        }
    }
    pub fn new_invalid() -> Self {
        Intersection {
            ray: Ray::new_invalid(),
            t: 0.0,
            point: Vec3::zero(),
            offset_by: 0.0,
            normal: Vec3::zero(),
        }
    }
}

#[derive(Clone, Copy)]
pub struct WIntersection {
    pub ray: WRay,
    pub t: f32x4,
    pub point: Wec3,
    pub offset_by: f32x4,
    pub normal: Wec3,
    pub basis: Wat3,
}

impl From<[Intersection; 4]> for WIntersection {
    fn from(intersections: [Intersection; 4]) -> Self {
        let ray = WRay::from([
            intersections[0].ray,
            intersections[0].ray,
            intersections[0].ray,
            intersections[0].ray,
        ]);
        let normal = Wec3::from([
            intersections[0].normal,
            intersections[1].normal,
            intersections[2].normal,
            intersections[3].normal,
        ]);
        let basis = normal.get_orthonormal_basis();
        let offset_by = f32x4::new(
            intersections[0].offset_by,
            intersections[1].offset_by,
            intersections[2].offset_by,
            intersections[3].offset_by,
        );
        let point = Wec3::from([
            intersections[0].point,
            intersections[1].point,
            intersections[2].point,
            intersections[3].point,
        ]);
        let t = f32x4::new(
            intersections[0].t,
            intersections[1].t,
            intersections[2].t,
            intersections[3].t,
        );

        Self {
            ray,
            t,
            point,
            offset_by,
            normal,
            basis,
        }
    }
}

impl WIntersection {
    pub fn new(ray: WRay, t: f32x4, point: Wec3, offset_by: f32x4, normal: Wec3) -> Self {
        WIntersection {
            ray,
            t,
            point,
            offset_by,
            normal,
            basis: normal.get_orthonormal_basis(),
        }
    }
    pub fn create_rays(&self, dir: Wec3) -> WRay {
        WRay::new(
            self.point + self.normal * self.normal.dot(dir).signum() * self.offset_by,
            dir,
            self.ray.time,
            self.ray.tile_coord,
        )
    }
}

pub trait Hitable: Send + Sync {
    fn hit(&self, rays: &WRay, t_ranges: ::std::ops::Range<f32x4>) -> f32x4;
    fn intersection_at(&self, ray: WRay, t: f32x4) -> (MaterialHandle, WIntersection);
}

pub struct HitStore<'bump> {
    hits: BumpVec<'bump, BumpVec<'bump, (Ray, f32)>>,
}

impl<'bump> HitStore<'bump> {
    pub fn from_material_store(bump: &'bump Bump, hitable_store: &HitableStore) -> Self {
        let mut hits = BumpVec::with_capacity_in(hitable_store.len(), bump);
        for _ in 0..hitable_store.len() {
            hits.push(BumpVec::new_in(bump))
        }
        Self { hits }
    }

    pub unsafe fn add_hit(&mut self, obj_id: usize, ray: Ray, t: f32) {
        self.hits.get_unchecked_mut(obj_id).push((ray, t));
    }

    pub fn process_hits(
        &mut self,
        hitables: &HitableStore,
        wintersections: &mut BumpVec<'_, (MaterialHandle, WIntersection)>,
    ) {
        let total_hits = self
            .hits
            .iter_mut()
            .map(|hits| {
                while hits.len() % 4 != 0 {
                    hits.push((Ray::new_invalid(), 0.0))
                }
                hits.len()
            })
            .sum::<usize>();

        wintersections.reserve(total_hits / 4);

        for (obj_id, hits) in self.hits.iter_mut().enumerate() {
            for hits in hits[0..].chunks(4) {
                // Safe because we just assured that every window will have exactly
                // 4 members.
                let hits: [(Ray, f32); 4] = unsafe {
                    [
                        *hits.get_unchecked(0),
                        *hits.get_unchecked(1),
                        *hits.get_unchecked(2),
                        *hits.get_unchecked(3),
                    ]
                };
                let rays = WRay::from([hits[0].0, hits[1].0, hits[2].0, hits[3].0]);
                let ts = f32x4::from([hits[0].1, hits[1].1, hits[2].1, hits[3].1]);
                wintersections
                    .push(unsafe { hitables.get_unchecked(obj_id) }.intersection_at(rays, ts));
            }
        }
    }

    pub fn reset(&mut self) {
        for isecs in self.hits.iter_mut() {
            isecs.clear();
        }
    }
}

pub struct HitableStore(Vec<Box<dyn Hitable>>);

impl HitableStore {
    pub fn new() -> Self {
        HitableStore(Vec::new())
    }

    pub fn push<H: Hitable + 'static>(&mut self, hitable: H) {
        self.0.push(Box::new(hitable))
    }
}

impl ::std::ops::Deref for HitableStore {
    type Target = Vec<Box<dyn Hitable>>;

    fn deref(&self) -> &Vec<Box<dyn Hitable>> {
        &self.0
    }
}

impl HitableStore {
    pub fn add_hits(
        &self,
        rays: WRay,
        t_ranges: ::std::ops::Range<f32x4>,
        hit_store: &mut HitStore,
    ) {
        let (ids, dists) = self.iter().enumerate().fold(
            ([std::usize::MAX; 4], t_ranges.end),
            |acc, (hitable_id, hitable)| {
                let (mut closest_ids, mut closest) = acc;

                let t = hitable.hit(&rays, t_ranges.start..closest);

                for ((t, closest), closest_id) in t
                    .as_ref()
                    .iter()
                    .zip(closest.as_mut().iter_mut())
                    .zip(closest_ids.iter_mut())
                {
                    if *t < *closest {
                        *closest = *t;
                        *closest_id = hitable_id;
                    }
                }

                (closest_ids, closest)
            },
        );

        let points: [Vec3; 4] = rays.point_at(dists).into();
        let rays: [Ray; 4] = rays.into();
        let dists = dists.as_ref();

        for (((id, point), ray), t) in ids
            .iter()
            .zip(points.iter())
            .zip(rays.iter())
            .zip(dists.iter())
        {
            if *id < std::usize::MAX {
                unsafe {
                    hit_store.add_hit(*id, *ray, *t);
                }
            }
        }
    }
}
