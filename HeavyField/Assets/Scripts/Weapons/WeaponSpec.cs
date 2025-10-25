using System;

namespace HeavyField.Weapons
{
    [Serializable]
    public class WeaponSpec
    {
        public string id;
        public string displayName;
        public string category;
        public float fireRate;
        public int magazineSize;
        public int reserveAmmo;
        public float reloadTime;
        public float damage;
        public float projectileSpeed;
        public bool automatic;
        public float spread;
    }
}
