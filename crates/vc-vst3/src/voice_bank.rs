//! Four independently loaded voice slots. Only the selected slot runs inference.
pub(crate) const LABELS: [&str; 4] = ["Toddler", "Girl", "Deep male", "Robot"];

pub(crate) struct VoiceBank<T> {
    slots: [Option<T>; 4],
    active: usize,
}
impl<T> VoiceBank<T> {
    pub(crate) fn new(slots: [Option<T>; 4], requested: usize) -> Option<Self> {
        let active = if slots.get(requested).is_some_and(Option::is_some) {
            requested
        } else {
            slots.iter().position(Option::is_some)?
        };
        Some(Self { slots, active })
    }
    pub(crate) fn active(&self) -> usize { self.active }
    pub(crate) fn insert(&mut self, slot: usize, value:T) -> anyhow::Result<()> {
        let target=self.slots.get_mut(slot).ok_or_else(||anyhow::anyhow!("Invalid voice slot"))?;
        anyhow::ensure!(target.is_none(),"Voice slot already loaded");
        *target=Some(value); Ok(())
    }
    pub(crate) fn available_mask(&self)->u32 { self.slots.iter().enumerate().fold(0,|mask,(i,slot)|if slot.is_some(){mask|(1u32<<(i+1))}else{mask}) }
    pub(crate) fn select(&mut self, slot: usize) -> anyhow::Result<bool> {
        if !self.slots.get(slot).is_some_and(Option::is_some) {
            anyhow::bail!("Voice slot {} is not loaded", slot + 1);
        }
        let changed = slot != self.active;
        self.active = slot;
        Ok(changed)
    }
    pub(crate) fn current(&self) -> &T { self.slots[self.active].as_ref().expect("active voice") }
    pub(crate) fn current_mut(&mut self) -> &mut T { self.slots[self.active].as_mut().expect("active voice") }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn incrementally_loaded_slot_preserves_active_model() {
        let mut b=VoiceBank::new([None,Some(2),None,None],1).unwrap();
        assert_eq!(b.available_mask(),4);
        b.insert(0,1).unwrap();
        assert_eq!(b.available_mask(),6);
        assert_eq!(*b.current(),2);
        assert!(b.select(0).unwrap());
        assert_eq!(*b.current(),1);
        assert!(b.insert(0,99).is_err());
    }
    #[test]
    fn all_four_slots_switch_without_reloading() {
        let mut b = VoiceBank::new([Some(1), Some(2), Some(3), Some(4)], 0).unwrap();
        *b.current_mut() = 11;
        for i in 1..4 { assert!(b.select(i).unwrap()); assert_eq!(*b.current(), i+1); }
        assert!(b.select(0).unwrap());
        assert_eq!(*b.current(), 11);
        assert!(!b.select(0).unwrap());
    }
    #[test]
    fn unavailable_voice_does_not_displace_current_voice() {
        let mut b = VoiceBank::new([None, Some(2), None, Some(4)], 0).unwrap();
        assert_eq!(b.active(), 1);
        assert!(b.select(0).is_err());
        assert!(b.select(99).is_err());
        assert_eq!(*b.current(), 2);
        assert!(VoiceBank::<i32>::new([None,None,None,None], 0).is_none());
    }
}
