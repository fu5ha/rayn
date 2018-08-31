extern crate vek;
extern crate image;

type Vec3 = vek::vec::repr_c::Vec3<f32>;
type Vec2 = vek::vec::repr_c::Vec2<f32>;
struct Color(Vec3);

impl From<Color> for image::Rgb<u8> {
    fn from(col: Color) -> Self {
        image::Rgb {
            data: [(col.0.x * 255.0) as u8, (col.0.y * 255.0) as u8, (col.0.z * 255.0) as u8]
        }
    }
}

const DIMS: (f32, f32) = (960.0, 540.0);
// const DIMS: (f32, f32) = (100.0, 50.0);

struct Ray {
    orig: Vec3,
    dir: Vec3,
}

impl Ray {
    pub fn new(orig: Vec3, dir: Vec3) -> Self { Ray { orig, dir } }
    pub fn orig(&self) -> &Vec3 { &self.orig }
    pub fn dir(&self) -> &Vec3 { &self.dir }
    pub fn point_at(&self, t: f32) -> Vec3 {
        self.dir.mul_add(Vec3::new(t,t,t), self.orig)
    }
}

#[derive(Clone, Copy, Debug)]
struct HitRecord {
    t: f32,
    p: Vec3,
    n: Vec3,
}

impl HitRecord {
    pub fn new(t: f32, p: Vec3, n: Vec3) -> Self { HitRecord { t, p, n } }
}

trait Hitable {
    fn hit(&self, ray: &Ray, t_range: ::std::ops::Range<f32>) -> Option<HitRecord>;
}

struct Sphere {
    orig: Vec3,
    rad: f32,
}

impl Sphere {
    pub fn new(orig: Vec3, rad: f32) -> Self { Sphere { orig, rad } }
    pub fn orig(&self) -> &Vec3 { &self.orig }
}

struct HitableList(Vec<Box<Hitable>>);

impl HitableList {
    pub fn new() -> Self { HitableList(Vec::new()) }

    pub fn push(&mut self, hitable: Box<Hitable>) {
        self.0.push(hitable)
    }
}

impl ::std::ops::Deref for HitableList {
    type Target = Vec<Box<Hitable>>;

    fn deref(&self) -> &Vec<Box<Hitable>> { &self.0 }
}

impl Hitable for HitableList {
    fn hit(&self, ray: &Ray, t_range: ::std::ops::Range<f32>) -> Option<HitRecord> {
        // print!("| ");
        let ret = self.iter()
            .fold((None, t_range.end), |acc, hitable| {
                let mut closest = acc.1;
                // print!("{} ", closest);
                let hr = hitable.hit(ray, t_range.start..closest);
                // print!("{:?} | ", hr);
                if let Some(HitRecord{t, p: _, n: _}) = hr {
                    closest = t;
                }
                let hr = if hr.is_some() { hr } else { acc.0 };
                (hr, closest)
            })
            .0;
        // println!(" | ");
        ret
    }
}

impl Hitable for Sphere {
    fn hit(&self, ray: &Ray, t_range: ::std::ops::Range<f32>) -> Option<HitRecord> {
        let oc = ray.orig() - self.orig;
        let a = ray.dir().dot(ray.dir().clone());
        let b = 2.0 * oc.clone().dot(ray.dir().clone());
        let c = oc.clone().dot(oc) - self.rad * self.rad;
        let descrim = b*b - 4.0*a*c;

        if descrim >= 0.0 {
            let desc_sqrt = descrim.sqrt();
            let t = (-b - desc_sqrt) / (2.0 * a);
            if t > t_range.start && t < t_range.end {
                let p = ray.point_at(t);
                let mut n = p - self.orig();
                n /= self.rad;
                return Some(HitRecord::new(t, p, n));
            }
            let t = (-b + desc_sqrt) / (2.0 * a);
            if t > t_range.start && t < t_range.end {
                let p = ray.point_at(t);
                let mut n = p - self.orig();
                n /= self.rad;
                return Some(HitRecord::new(t, p, n));
            }
        }
        None
    }
}

fn compute_color(ray: &Ray, hitables: &HitableList) -> Color {
    if let Some(record) = hitables.hit(ray, 0.0..100.0) {
        Color(Vec3::from(0.5) * (Vec3::one() + record.n))
    } else {
        let mut dir = ray.dir().clone();
        dir.normalize();
        let t = 0.5 * (dir.y + 1.0);

        Color(Vec3::lerp(Vec3::one(), Vec3::new(0.5, 0.7, 1.0), t))
    }
}

fn main() {
    let mut img = image::RgbImage::new(DIMS.0 as u32, DIMS.1 as u32);

    let top_left = Vec3::new(-DIMS.0 / DIMS.1, 1.0, -1.0);
    let view_full = Vec3::new(DIMS.0 / DIMS.1 * 2.0, -2.0, 0.0);
    let origin = Vec3::new(0.0, 0.0, 0.0);

    let mut world = HitableList::new();
    world.push(Box::new(Sphere::new(Vec3::new(0.0, -100.5, -1.0), 100.0)));
    world.push(Box::new(Sphere::new(Vec3::new(0.0, 0.0, -1.0), 0.5)));

    for (x, y, pixel) in img.enumerate_pixels_mut() {
        // print!("({}, {}) ", x, y);
        let uv = Vec3::new(x as f32 / DIMS.0 as f32, y as f32 / DIMS.1 as f32, 0.0);
        let ray = Ray::new(origin.clone(), top_left + (view_full * uv));
        let col = compute_color(&ray, &world);
        *pixel = col.into();
    }

    img.save("render.png").unwrap();

}
