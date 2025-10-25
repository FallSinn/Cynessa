using System.Collections.Generic;
using UnityEngine;

namespace HeavyField.Weapons
{
    public class WeaponManager : MonoBehaviour
    {
        [SerializeField] private WeaponDatabase database;
        [SerializeField] private List<WeaponBehaviour> weaponSlots = new List<WeaponBehaviour>();
        [SerializeField] private float reloadCooldown = 0.1f;

        private int currentIndex;
        private float reloadEndTime;

        public WeaponBehaviour CurrentWeapon => weaponSlots.Count > 0 ? weaponSlots[currentIndex] : null;
        public WeaponSpec CurrentSpec => CurrentWeapon?.Spec;
        public int CurrentAmmo => CurrentWeapon?.CurrentAmmo ?? 0;
        public int CurrentReserve => CurrentWeapon?.CurrentReserve ?? 0;

        private void Start()
        {
            InitializeWeapons();
        }

        private void InitializeWeapons()
        {
            foreach (WeaponBehaviour weapon in weaponSlots)
            {
                WeaponSpec spec = database.GetById(weapon.WeaponId);
                if (spec != null)
                {
                    weapon.Initialize(spec);
                }
                else
                {
                    Debug.LogWarning($"Weapon ID {weapon.WeaponId} not found.");
                }
            }

            SelectWeapon(0);
        }

        public void TryFireCurrent()
        {
            if (!CurrentWeapon)
            {
                return;
            }

            if (Time.time < reloadEndTime)
            {
                return;
            }

            bool fired = CurrentWeapon.TryFire();
            if (!fired && CurrentWeapon.CurrentAmmo <= 0)
            {
                ReloadCurrent();
            }
        }

        public void ReloadCurrent()
        {
            if (!CurrentWeapon)
            {
                return;
            }

            reloadEndTime = Time.time + CurrentWeapon.Spec.reloadTime + reloadCooldown;
            Invoke(nameof(FinishReload), CurrentWeapon.Spec.reloadTime);
        }

        private void FinishReload()
        {
            CurrentWeapon?.Reload();
        }

        public void NextWeapon()
        {
            if (weaponSlots.Count <= 1)
            {
                return;
            }

            int next = (currentIndex + 1) % weaponSlots.Count;
            SelectWeapon(next);
        }

        public void PreviousWeapon()
        {
            if (weaponSlots.Count <= 1)
            {
                return;
            }

            int prev = (currentIndex - 1 + weaponSlots.Count) % weaponSlots.Count;
            SelectWeapon(prev);
        }

        private void SelectWeapon(int index)
        {
            currentIndex = index;
            for (int i = 0; i < weaponSlots.Count; i++)
            {
                bool active = i == currentIndex;
                if (weaponSlots[i])
                {
                    weaponSlots[i].gameObject.SetActive(active);
                }
            }
        }

        public void SetAim(bool aiming)
        {
            if (CurrentWeapon)
            {
                CurrentWeapon.SetAim(aiming);
            }
        }
    }
}
