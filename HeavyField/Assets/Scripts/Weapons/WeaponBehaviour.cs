using UnityEngine;
using HeavyField.Systems;

namespace HeavyField.Weapons
{
    public class WeaponBehaviour : MonoBehaviour
    {
        [SerializeField] private Transform muzzlePoint;
        [SerializeField] private ParticleSystem muzzleFlash;
        [SerializeField] private string weaponId;

        private WeaponSpec spec;
        private float nextFireTime;
        private int currentAmmo;
        private int currentReserve;
        private bool isAiming;

        public string WeaponId => weaponId;
        public WeaponSpec Spec => spec;
        public int CurrentAmmo => currentAmmo;
        public int CurrentReserve => currentReserve;

        public void Initialize(WeaponSpec weaponSpec)
        {
            spec = weaponSpec;
            currentAmmo = spec.magazineSize;
            currentReserve = spec.reserveAmmo;
        }

        public void SetAim(bool aiming)
        {
            isAiming = aiming;
        }

        public bool TryFire()
        {
            if (spec == null)
            {
                Debug.LogWarning("Weapon spec not initialized.");
                return false;
            }

            if (Time.time < nextFireTime)
            {
                return false;
            }

            if (currentAmmo <= 0)
            {
                return false;
            }

            if (!spec.automatic && Input.GetButton("Fire1") && Time.time < nextFireTime + 0.01f)
            {
                return false;
            }

            FireProjectile();
            currentAmmo--;
            nextFireTime = Time.time + 1f / Mathf.Max(0.01f, spec.fireRate);
            return true;
        }

        private void FireProjectile()
        {
            if (muzzleFlash)
            {
                muzzleFlash.Play();
            }

            Bullet projectile = ObjectPooler.Instance.SpawnFromPool("Bullet", muzzlePoint.position, GetSpreadRotation());
            if (projectile)
            {
                projectile.Configure(spec.damage, spec.projectileSpeed, gameObject.layer);
            }
        }

        private Quaternion GetSpreadRotation()
        {
            Vector3 forward = muzzlePoint.forward;
            float spreadAmount = isAiming ? spec.spread * 0.5f : spec.spread;
            Vector3 random = Vector3.zero;
            if (spreadAmount > 0f)
            {
                random = Random.insideUnitCircle * spreadAmount * Mathf.Deg2Rad;
            }

            Quaternion offset = Quaternion.Euler(random.x, random.y, 0f);
            return offset * Quaternion.LookRotation(forward);
        }

        public void Reload()
        {
            if (currentAmmo >= spec.magazineSize)
            {
                return;
            }

            if (currentReserve <= 0)
            {
                return;
            }

            int missing = spec.magazineSize - currentAmmo;
            int toLoad = Mathf.Min(missing, currentReserve);
            currentAmmo += toLoad;
            currentReserve -= toLoad;
        }

        public void FillReserve(int amount)
        {
            currentReserve += amount;
        }
    }
}
