using UnityEngine;
using HeavyField.Player;
using HeavyField.Weapons;

namespace HeavyField.Systems
{
    public class Pickup : MonoBehaviour
    {
        public enum PickupType
        {
            Health,
            Ammo
        }

        [SerializeField] private PickupType type;
        [SerializeField] private float healthAmount = 25f;
        [SerializeField] private int ammoAmount = 30;

        private void OnTriggerEnter(Collider other)
        {
            if (type == PickupType.Health && other.TryGetComponent(out PlayerHealth health))
            {
                health.Heal(healthAmount);
                Destroy(gameObject);
            }
            else if (type == PickupType.Ammo)
            {
                WeaponManager manager = other.GetComponentInChildren<WeaponManager>();
                if (manager && manager.CurrentWeapon)
                {
                    manager.CurrentWeapon.FillReserve(ammoAmount);
                    Destroy(gameObject);
                }
            }
        }
    }
}
