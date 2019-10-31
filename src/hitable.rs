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
    pub valid: bool,
}

impl Intersection {
    pub fn new(ray: Ray, t: f32, point: Vec3, offset_by: f32, normal: Vec3) -> Self {
        Intersection {
            ray,
            t,
            point,
            offset_by,
            normal,
            valid: true,
        }
    }
    pub fn new_invalid() -> Self {
        Intersection {
            ray: Ray::new(Vec3::zero(), Vec3::zero(), 0.0, Vec2u::zero()),
            t: 0.0,
            point: Vec3::zero(),
            offset_by: 0.0,
            normal: Vec3::zero(),
            valid: false,
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
    pub valid: [bool; 4],
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

        let valid = [
            intersections[0].valid,
            intersections[1].valid,
            intersections[2].valid,
            intersections[3].valid,
        ];

        Self {
            ray,
            t,
            point,
            offset_by,
            normal,
            basis,
            valid,
        }
    }
}

impl WIntersection {
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
    fn intersection_at(&self, ray: Ray, t: f32, point: Vec3) -> (MaterialHandle, Intersection);
}

pub struct HitStore<'bump> {
    hits: BumpVec<'bump, BumpVec<'bump, Intersection>>,
}

impl<'bump> HitStore<'bump> {
    pub fn from_material_store(bump: &'bump Bump, mat_store: &MaterialStore) -> Self {
        let mut hits = BumpVec::with_capacity_in(mat_store.len(), bump);
        for _ in 0..mat_store.len() {
            hits.push(BumpVec::new_in(bump))
        }
        Self { hits }
    }

    pub fn add_hit(&mut self, mat_id: MaterialHandle, isec: Intersection) {
        self.hits[mat_id.0].push(isec);
    }

    pub fn prepare_wintersections(
        &mut self,
        wintersections: &mut BumpVec<'_, (MaterialHandle, WIntersection)>,
    ) {
        let total_isecs = self
            .hits
            .iter_mut()
            .map(|isecs| {
                while isecs.len() % 4 != 0 {
                    isecs.push(Intersection::new_invalid())
                }
                isecs.len()
            })
            .sum::<usize>();

        wintersections.reserve(total_isecs / 4);

        for (mat_id, isecs) in self.hits.iter_mut().enumerate() {
            for isecs in isecs[0..].chunks(4) {
                wintersections.push((
                    MaterialHandle(mat_id),
                    // Safe because we just assured that every window will have exactly
                    // 4 members.
                    WIntersection::from(unsafe {
                        [
                            *isecs.get_unchecked(0),
                            *isecs.get_unchecked(1),
                            *isecs.get_unchecked(2),
                            *isecs.get_unchecked(3),
                        ]
                    }),
                ));
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
                let (mat, isec) = self[*id].intersection_at(*ray, *t, *point);
                hit_store.add_hit(mat, isec);
            }
        }
    }
}
