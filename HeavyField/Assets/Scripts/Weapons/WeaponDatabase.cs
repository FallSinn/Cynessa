using System.Collections.Generic;
using UnityEngine;

namespace HeavyField.Weapons
{
    public class WeaponDatabase : MonoBehaviour
    {
        [SerializeField] private TextAsset weaponJson;

        private readonly Dictionary<string, WeaponSpec> lookup = new Dictionary<string, WeaponSpec>();
        public IReadOnlyDictionary<string, WeaponSpec> Specs => lookup;

        private void Awake()
        {
            if (!weaponJson)
            {
                Debug.LogError("WeaponDatabase requires a TextAsset json file.");
                return;
            }

            WeaponSpec[] specs = JsonUtility.FromJson<Wrapper>(weaponJson.text).weapons;
            lookup.Clear();
            foreach (WeaponSpec spec in specs)
            {
                if (!lookup.ContainsKey(spec.id))
                {
                    lookup.Add(spec.id, spec);
                }
            }
        }

        public WeaponSpec GetById(string id)
        {
            lookup.TryGetValue(id, out WeaponSpec spec);
            return spec;
        }

        [System.Serializable]
        private class Wrapper
        {
            public WeaponSpec[] weapons;
        }
    }
}
