using System.Collections.Generic;
using UnityEngine;

namespace HeavyField.Environment
{
    public class MapStreamer : MonoBehaviour
    {
        [System.Serializable]
        public class ChunkDefinition
        {
            public string name;
            public Vector2Int coordinates;
            public GameObject prefab;
        }

        [SerializeField] private Transform player;
        [SerializeField] private Vector2 chunkSize = new Vector2(100f, 100f);
        [SerializeField] private int viewDistance = 1;
        [SerializeField] private ChunkDefinition[] chunks;

        private readonly Dictionary<Vector2Int, GameObject> activeChunks = new Dictionary<Vector2Int, GameObject>();

        private void Update()
        {
            if (!player)
            {
                return;
            }

            Vector2 playerPos = new Vector2(player.position.x, player.position.z);
            Vector2Int currentChunk = WorldToChunk(playerPos);

            foreach (Vector2Int coord in new List<Vector2Int>(activeChunks.Keys))
            {
                if (Vector2Int.Distance(coord, currentChunk) > viewDistance)
                {
                    Destroy(activeChunks[coord]);
                    activeChunks.Remove(coord);
                }
            }

            for (int x = -viewDistance; x <= viewDistance; x++)
            {
                for (int y = -viewDistance; y <= viewDistance; y++)
                {
                    Vector2Int coord = new Vector2Int(currentChunk.x + x, currentChunk.y + y);
                    if (!activeChunks.ContainsKey(coord))
                    {
                        SpawnChunk(coord);
                    }
                }
            }
        }

        private void SpawnChunk(Vector2Int coord)
        {
            ChunkDefinition def = FindChunk(coord);
            if (def == null || def.prefab == null)
            {
                return;
            }

            Vector3 position = new Vector3(coord.x * chunkSize.x, 0f, coord.y * chunkSize.y);
            GameObject instance = Instantiate(def.prefab, position, Quaternion.identity, transform);
            activeChunks.Add(coord, instance);
        }

        private ChunkDefinition FindChunk(Vector2Int coord)
        {
            foreach (ChunkDefinition def in chunks)
            {
                if (def.coordinates == coord)
                {
                    return def;
                }
            }

            return null;
        }

        private Vector2Int WorldToChunk(Vector2 worldPosition)
        {
            int x = Mathf.RoundToInt(worldPosition.x / Mathf.Max(1f, chunkSize.x));
            int y = Mathf.RoundToInt(worldPosition.y / Mathf.Max(1f, chunkSize.y));
            return new Vector2Int(x, y);
        }
    }
}
