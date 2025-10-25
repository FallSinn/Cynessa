using UnityEngine;
using UnityEngine.UI;

namespace HeavyField.Systems
{
    public class HUDController : MonoBehaviour
    {
        [SerializeField] private Text healthText;
        [SerializeField] private Text ammoText;
        [SerializeField] private Text weaponText;

        public void UpdateHealth(float value)
        {
            if (healthText)
            {
                healthText.text = $"HP: {Mathf.RoundToInt(value)}";
            }
        }

        public void UpdateAmmo(int clip, int reserve)
        {
            if (ammoText)
            {
                ammoText.text = $"Ammo: {clip}/{reserve}";
            }
        }

        public void UpdateWeaponName(string name)
        {
            if (weaponText)
            {
                weaponText.text = name;
            }
        }
    }
}
