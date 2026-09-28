import Std

namespace ContrastCapacity

def Feasible (r k h q : Nat) : Prop := h ≤ r ∧ q ≤ k ∧ h + q = k

def capacity (r k : Nat) : Nat := min r k

theorem feasible_upper {r k h q : Nat} (hf : Feasible r k h q) :
    h ≤ capacity r k := by
  unfold Feasible at hf
  unfold capacity
  omega

theorem occupancy_iff (r k h : Nat) :
    (∃ q, Feasible r k h q) ↔ h ≤ capacity r k := by
  constructor
  · rintro ⟨q, hq⟩
    exact feasible_upper hq
  · intro hh
    refine ⟨k - h, ?_⟩
    unfold capacity at hh
    unfold Feasible
    omega

theorem capacity_attained (r k : Nat) :
    Feasible r k (capacity r k) (k - capacity r k) := by
  unfold Feasible capacity
  omega

theorem zero_occupancy_unique (r k q : Nat) :
    Feasible r k 0 q ↔ q = k := by
  unfold Feasible
  omega

theorem capacity_zero_iff (r k : Nat) :
    capacity r k = 0 ↔ r = 0 ∨ k = 0 := by
  unfold capacity
  omega

def total (f : Nat → Nat) : Nat → Nat
  | 0 => 0
  | n + 1 => total f n + f n

theorem total_mono (n : Nat) {f g : Nat → Nat}
    (h : ∀ i, i < n → f i ≤ g i) : total f n ≤ total g n := by
  induction n with
  | zero => simp [total]
  | succ n ih =>
    have hp := ih (fun i hi => h i (by omega))
    have hn := h n (by omega)
    simp only [total]
    omega

theorem total_capacity_upper (n : Nat) (r k h q : Nat → Nat)
    (hf : ∀ i, i < n → Feasible (r i) (k i) (h i) (q i)) :
    total h n ≤ total (fun i => capacity (r i) (k i)) n := by
  exact total_mono n (fun i hi => feasible_upper (hf i hi))

theorem total_capacity_attained (n : Nat) (r k : Nat → Nat) :
    ∃ h q : Nat → Nat,
      (∀ i, i < n → Feasible (r i) (k i) (h i) (q i)) ∧
      total h n = total (fun i => capacity (r i) (k i)) n := by
  refine ⟨(fun i => capacity (r i) (k i)),
    (fun i => k i - capacity (r i) (k i)), ?_, rfl⟩
  intro i _
  exact capacity_attained (r i) (k i)

theorem merged_capacity_upper (n : Nat) (r k : Nat → Nat) :
    total (fun i => capacity (r i) (k i)) n ≤
      capacity (total r n) (total k n) := by
  have hr : total (fun i => capacity (r i) (k i)) n ≤ total r n := by
    apply total_mono
    intro i _
    unfold capacity
    omega
  have hk : total (fun i => capacity (r i) (k i)) n ≤ total k n := by
    apply total_mono
    intro i _
    unfold capacity
    omega
  exact Nat.le_min.mpr ⟨hr, hk⟩


theorem refinement_capacity (m : Nat) (sizes : Nat → Nat)
    (r k : Nat → Nat → Nat) :
    total (fun j => total (fun i => capacity (r j i) (k j i)) (sizes j)) m ≤
    total (fun j => capacity (total (r j) (sizes j))
      (total (k j) (sizes j))) m := by
  apply total_mono
  intro j _
  exact merged_capacity_upper (sizes j) (r j) (k j)

theorem target_impossible (n target : Nat) (r k h q : Nat → Nat)
    (hf : ∀ i, i < n → Feasible (r i) (k i) (h i) (q i))
    (ht : total (fun i => capacity (r i) (k i)) n < target) :
    total h n < target := by
  have hu := total_capacity_upper n r k h q hf
  omega


theorem ratio_target_impossible (n a b : Nat) (r k h q : Nat → Nat)
    (hf : ∀ i, i < n → Feasible (r i) (k i) (h i) (q i))
    (ht : total (fun i => capacity (r i) (k i)) n * b < a * total r n) :
    total h n * b < a * total r n := by
  have hu := total_capacity_upper n r k h q hf
  exact Nat.lt_of_le_of_lt (Nat.mul_le_mul_right b hu) ht

end ContrastCapacity

#print axioms ContrastCapacity.feasible_upper
#print axioms ContrastCapacity.occupancy_iff
#print axioms ContrastCapacity.capacity_attained
#print axioms ContrastCapacity.zero_occupancy_unique
#print axioms ContrastCapacity.capacity_zero_iff
#print axioms ContrastCapacity.total_mono
#print axioms ContrastCapacity.total_capacity_upper
#print axioms ContrastCapacity.total_capacity_attained
#print axioms ContrastCapacity.merged_capacity_upper
#print axioms ContrastCapacity.refinement_capacity
#print axioms ContrastCapacity.target_impossible
#print axioms ContrastCapacity.ratio_target_impossible

namespace ContrastCapacity


theorem expectation_between_half_capacity_and_capacity (r k : Nat) :
    capacity r k * (r+k) ≤ 2*(r*k) ∧
    r*k ≤ capacity r k * (r+k) := by
  by_cases h : r ≤ k
  · have hm : capacity r k = r := Nat.min_eq_left h
    rw [hm]
    constructor
    · have hs : r+k ≤ k+k := Nat.add_le_add_right h k
      have hp := Nat.mul_le_mul_left r hs
      simpa [Nat.mul_add, Nat.two_mul] using hp
    · exact Nat.mul_le_mul_left r (by omega)
  · have hk : k ≤ r := by omega
    have hm : capacity r k = k := Nat.min_eq_right hk
    rw [hm]
    constructor
    · calc
        k * (r+k) = r*k + k*k := by rw [Nat.mul_add, Nat.mul_comm k r]
        _ ≤ r*k + r*k := Nat.add_le_add_left (Nat.mul_le_mul_right k hk) (r*k)
        _ = 2*(r*k) := (Nat.two_mul (r*k)).symm
    · have hp := Nat.mul_le_mul_left k (show r ≤ r+k by omega)
      simpa [Nat.mul_comm] using hp

theorem total_zero_iff (n : Nat) (f : Nat → Nat) :
    total f n = 0 ↔ ∀ i, i < n → f i = 0 := by
  induction n with
  | zero => simp [total]
  | succ n ih =>
    constructor
    · intro hz
      have hs : total f n + f n = 0 := hz
      have hp : total f n = 0 := by omega
      have hn : f n = 0 := by omega
      intro i hi
      by_cases he : i = n
      · simpa [he] using hn
      · exact ih.mp hp i (by omega)
    · intro hall
      have hp := ih.mpr (fun i hi => hall i (by omega))
      have hn := hall n (by omega)
      simp [total, hp, hn]

theorem total_binary_le (n : Nat) (f : Nat → Nat)
    (hb : ∀ i, i < n → f i ≤ 1) : total f n ≤ n := by
  induction n with
  | zero => simp [total]
  | succ n ih =>
    have hp := ih (fun i hi => hb i (by omega))
    have hn := hb n (by omega)
    simp only [total]
    omega

theorem total_binary_full (n : Nat) (f : Nat → Nat)
    (hb : ∀ i, i < n → f i ≤ 1) (ht : total f n = n) :
    ∀ i, i < n → f i = 1 := by
  induction n with
  | zero => intro i hi; omega
  | succ n ih =>
    have hb' : ∀ i, i < n → f i ≤ 1 := fun i hi => hb i (by omega)
    have hle := total_binary_le n f hb'
    have hnle := hb n (by omega)
    have hs : total f n + f n = n+1 := ht
    have hp : total f n = n := by omega
    have hn : f n = 1 := by omega
    intro i hi
    by_cases he : i = n
    · simpa [he] using hn
    · exact ih hb' hp i (by omega)


theorem zero_selection_is_deletion (r k : Nat) (h o : Nat → Nat)
    (ho : ∀ i, i < k → o i ≤ 1)
    (size : total h r + total o k = k)
    (zero : total h r = 0) :
    (∀ i, i < r → h i = 0) ∧ (∀ i, i < k → o i = 1) := by
  constructor
  · exact (total_zero_iff r h).mp zero
  · apply total_binary_full k o ho
    omega

end ContrastCapacity

#print axioms ContrastCapacity.expectation_between_half_capacity_and_capacity
#print axioms ContrastCapacity.total_zero_iff
#print axioms ContrastCapacity.total_binary_le
#print axioms ContrastCapacity.total_binary_full
#print axioms ContrastCapacity.zero_selection_is_deletion
