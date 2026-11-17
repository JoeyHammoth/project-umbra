use crate::object::AttenuationObject;

/// A collection of attenuation objects.  The attenuation at a given point is
/// equal to the sum of the attenuations of all the objects within the
/// collection.
///
/// Although a negative attenuation doesn't make much sense for standard
/// objects, this can be used in an object collection to create a region of
/// lesser attenuation within an object with larger attenuation.
pub struct ObjectCollection<'a> {
    objects: Vec<&'a (dyn AttenuationObject + 'a)>,
}

impl<'a> ObjectCollection<'a> {
    /// Create a new empty object collection.
    ///
    /// # Example
    ///
    /// ```
    /// use ctlab::object::ObjectCollection;
    ///
    /// let collection = ObjectCollection::new();
    /// ```
    pub fn new() -> ObjectCollection<'a> {
        ObjectCollection {
            objects: Vec::new(),
        }
    }

    /// Add an object to the collection.
    ///
    /// # Example
    ///
    /// ```
    /// use ctlab::object::{
    ///     ObjectCollection,
    ///     Circle,
    ///     Rectangle};
    /// use ctlab::object::AttenuationObject;
    ///
    /// let circle = Circle::new(0., 0., 1., 3.);
    /// let rect = Rectangle::new((-3., 3.), (-2., 1.), 3.);
    /// let mut collection = ObjectCollection::new();
    /// collection.push(&circle);
    /// collection.push(&rect);
    ///
    /// assert_eq!(
    ///     collection.project_attenuation(0., 0., (-10., 10.)),
    ///     circle.project_attenuation(0., 0., (-10., 10.))
    ///     + rect.project_attenuation(0., 0., (-10., 10.))
    /// )
    /// ```
    pub fn push<T: AttenuationObject>(&mut self, object: &'a T) {
        self.objects.push(object);
    }
}

impl<'a> AttenuationObject for ObjectCollection<'a> {
    fn attenuation(&self, x: f64, y: f64) -> f64 {
        self.objects.iter().map(|o| o.attenuation(x, y)).sum()
    }
}

impl<'a> Default for ObjectCollection<'a> {
    fn default() -> ObjectCollection<'a> {
        ObjectCollection::new()
    }
}

// Allow ObjectCollection to be synced across threads
unsafe impl<'a> Sync for ObjectCollection<'a> {}
