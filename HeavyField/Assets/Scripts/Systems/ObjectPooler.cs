using System.Collections.Generic;
using UnityEngine;

namespace HeavyField.Systems
{
    public class ObjectPooler : MonoBehaviour
    {
        [System.Serializable]
        private class Pool
        {
            public string key;
            public Bullet prefab;
            public int initialSize = 16;
        }

        [SerializeField] private Pool[] pools;

        private readonly Dictionary<string, Queue<Bullet>> poolLookup = new Dictionary<string, Queue<Bullet>>();
        public static ObjectPooler Instance { get; private set; }

        private void Awake()
        {
            if (Instance && Instance != this)
            {
                Destroy(gameObject);
                return;
            }

            Instance = this;

            foreach (Pool pool in pools)
            {
                Queue<Bullet> queue = new Queue<Bullet>();
                for (int i = 0; i < pool.initialSize; i++)
                {
                    Bullet bullet = Instantiate(pool.prefab, transform);
                    bullet.gameObject.SetActive(false);
                    queue.Enqueue(bullet);
                }

                poolLookup.Add(pool.key, queue);
            }
        }

        public Bullet SpawnFromPool(string key, Vector3 position, Quaternion rotation)
        {
            if (!poolLookup.TryGetValue(key, out Queue<Bullet> queue))
            {
                Debug.LogWarning($"No pool named {key} exists.");
                return null;
            }

            Bullet bullet = queue.Count > 0 ? queue.Dequeue() : null;
            if (!bullet)
            {
                Debug.LogWarning($"Pool {key} is empty and has no prefab fallback.");
                return null;
            }

            bullet.transform.SetPositionAndRotation(position, rotation);
            bullet.gameObject.SetActive(true);
            bullet.OnReturnedToPool = () => ReturnToPool(key, bullet);
            bullet.OnSpawned();
            return bullet;
        }

        private void ReturnToPool(string key, Bullet bullet)
        {
            bullet.gameObject.SetActive(false);
            poolLookup[key].Enqueue(bullet);
        }
    }
}
