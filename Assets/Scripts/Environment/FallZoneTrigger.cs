using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// Dipasang oleh GeometryBugMarker pada child "FallZone".
//
// Dua masalah pendekatan lama (OnTriggerExit + WaitForSeconds) yang sudah diperbaiki:
//   1. Sprint teleport: agen keluar trigger horizontal sebelum jatuh, restoration timer
//      jalan sementara agen masih overlap tile → depenetration dorong agen ke atas.
//      Fix: RestoreCollision hanya jalan setelah agen terkonfirmasi di bawah permukaan.
//   2. Double jump: Physics.IgnoreCollision tidak mempengaruhi Physics.Raycast,
//      sehingga grounded = true tetap terbaca → lompat tetap bisa dilakukan.
//      Fix: static set "activeAgents" dipakai agen untuk override grounded check.
public class FallZoneTrigger : MonoBehaviour
{
    [HideInInspector] public LayerMask groundLayer;
    [HideInInspector] public Vector3   bugPosition;

    // Agen yang sedang aktif dalam fall zone ini; diquery oleh agent scripts
    // untuk override grounded check (buat grounded = false saat di dalam lubang).
    public static HashSet<Collider> activeAgents = new HashSet<Collider>();

    public static bool IsInFallZone(Collider col) => activeAgents.Contains(col);

    // Pasangan collider agen → daftar ground tile yang sedang di-ignore
    private Dictionary<Collider, List<Collider>> ignoredPairs =
        new Dictionary<Collider, List<Collider>>();

    private void OnTriggerEnter(Collider other)
    {
        if (other.GetComponent<Rigidbody>() == null) return;
        if (ignoredPairs.ContainsKey(other)) return;

        // Cari ground tile tepat di bawah kaki agen dengan Raycast
        List<Collider> toIgnore = new List<Collider>();
        RaycastHit[] hits = Physics.RaycastAll(
            other.bounds.center, Vector3.down,
            other.bounds.extents.y + 2f, groundLayer
        );
        foreach (RaycastHit h in hits)
        {
            Physics.IgnoreCollision(other, h.collider, true);
            toIgnore.Add(h.collider);
        }

        // Fallback: OverlapSphere kecil di posisi kaki
        if (toIgnore.Count == 0)
        {
            Vector3 feet = new Vector3(other.bounds.center.x,
                                       other.bounds.min.y,
                                       other.bounds.center.z);
            foreach (Collider col in Physics.OverlapSphere(feet, 0.5f, groundLayer))
            {
                Physics.IgnoreCollision(other, col, true);
                toIgnore.Add(col);
            }
        }

        if (toIgnore.Count == 0) return;

        ignoredPairs[other] = toIgnore;
        activeAgents.Add(other);

        // Monitor posisi agen: restore hanya setelah terbukti di bawah lantai
        StartCoroutine(MonitorAndRestore(other, toIgnore));

        Debug.Log($"FallZone [{bugPosition}]: {other.name} masuk, " +
                  $"{toIgnore.Count} ground collider diabaikan.");
    }

    // OnTriggerExit tidak lagi dipakai untuk restoration.
    // Coroutine MonitorAndRestore yang mengontrol kapan collision dipulihkan.
    //private void OnTriggerExit(Collider other) { }

    private IEnumerator MonitorAndRestore(Collider agent, List<Collider> grounds)
    {
        float groundY  = bugPosition.y;
        float fallThreshold = groundY - 1.2f;   // agen dianggap sudah jatuh jika di sini
        float timeout  = 4f;
        float elapsed  = 0f;

        // Tunggu sampai agen benar-benar di bawah lantai, atau timeout
        while (elapsed < timeout)
        {
            if (agent == null) break;
            if (agent.transform.position.y < fallThreshold) break;
            elapsed += Time.deltaTime;
            yield return null;
        }

        // Beri jeda singkat agar agen sudah jauh dari tile sebelum collision aktif lagi,
        // mencegah depenetration mendorong agen ke atas
        yield return new WaitForSeconds(0.2f);

        // Pulihkan collision
        foreach (Collider g in grounds)
        {
            if (agent != null && g != null)
                Physics.IgnoreCollision(agent, g, false);
        }

        if (agent != null) activeAgents.Remove(agent);
        if (ignoredPairs.ContainsKey(agent)) ignoredPairs.Remove(agent);

        Debug.Log($"FallZone [{bugPosition}]: collision {agent?.name} dipulihkan " +
                  $"(elapsed={elapsed:F1}s, y={agent?.transform.position.y:F1}).");
    }

    private void OnDestroy()
    {
        // Bersihkan state statis saat objek dihancurkan (misal: scene reload)
        ignoredPairs.Clear();
        activeAgents.Clear();
    }
}
