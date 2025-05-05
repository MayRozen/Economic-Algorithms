import heapq
from collections import defaultdict


class EnvyGraph:
    def __init__(self):
        self.graph = defaultdict(list)  # מבנה הנתונים המייצג את הגרף
        self.nodes = set()  # קבוצת הקודקודים בגרף
        self.item_values = defaultdict(dict)  # ערכי החפצים לכל קודקוד {node: {item_id: value}}

    def add_node(self, node, item_id, own_value, values_for_others=None):
        """
        הוספת קודקוד עם ערך החפץ שברשותו וערכים עבור חפצים של אחרים

        Parameters:
        - node: מזהה הקודקוד
        - item_id: מזהה החפץ שבבעלות הקודקוד
        - own_value: הערך שהקודקוד מעניק לחפץ שברשותו
        - values_for_others: מילון של {node_id: value} המציין את הערך שהקודקוד מעניק לחפצים של אחרים
        """
        self.nodes.add(node)
        self.item_values[node][item_id] = own_value

        # הוספת ערכים עבור חפצים של קודקודים אחרים
        if values_for_others:
            for other_node, value in values_for_others.items():
                # שמירת הערך שהקודקוד מעניק לחפץ של אחר
                self.item_values[node][other_node] = value

    def calculate_envy(self, source, target):
        """
        חישוב רמת הקנאה של קודקוד מקור בקודקוד יעד
        הקנאה מוגדרת כהערכה של החפץ של היעד פחות הערכה של החפץ של המקור
        קנאה חיובית משמעותה שהמקור מעדיף את החפץ של היעד
        קנאה שלילית משמעותה שהמקור מעדיף את החפץ שלו
        """
        # מציאת החפץ של כל קודקוד (בהנחה שלכל קודקוד יש חפץ אחד משלו)
        source_item = None
        target_item = None

        # מציאת החפץ של המקור והיעד
        for item_id, value in self.item_values[source].items():
            if item_id == source:  # אם החפץ שייך למקור
                source_item = item_id
                break

        for item_id, value in self.item_values[target].items():
            if item_id == target:  # אם החפץ שייך ליעד
                target_item = item_id
                break

        if source_item is None or target_item is None:
            return 0  # אם אין חפצים, אין קנאה

        # ערך החפץ של היעד בעיני המקור
        target_value_for_source = self.item_values[source].get(target_item, 0)

        # ערך החפץ של המקור בעיני עצמו
        source_value_for_source = self.item_values[source].get(source_item, 0)

        # חישוב הקנאה: כמה המקור מעריך את חפץ היעד פחות כמה הוא מעריך את החפץ שלו
        envy = target_value_for_source - source_value_for_source

        return envy

    def build_envy_edges(self):
        """בניית צלעות הקנאה בגרף על סמך ערכי החפצים"""
        # איפוס הגרף הקיים
        self.graph = defaultdict(list)

        # עבור כל זוג קודקודים, חשב את הקנאה וצור צלע מתאימה
        for source in self.nodes:
            for target in self.nodes:
                if source != target:  # אין קנאה עצמית
                    envy = self.calculate_envy(source, target)
                    # הוספת צלע מכוונת עם משקל הקנאה
                    self.graph[source].append((target, envy))

    def add_envy_edge(self, source, target, envy_weight):
        """הוספת צלע מכוונת עם משקל קנאה באופן ישיר"""
        self.graph[source].append((target, envy_weight))
        self.nodes.add(source)
        self.nodes.add(target)

    def find_max_avg_weight_cycle(self):
        """מציאת המעגל עם המשקל הממוצע הגבוה ביותר"""
        if not self.nodes:
            return None, 0

        # בדיקה מיוחדת לצלעות עם משקל אפס
        for node in self.nodes:
            for neighbor, weight in self.graph[node]:
                if node == neighbor and weight == 0:
                    return [node, node], 0

        max_avg = float('-inf')
        max_cycle = None

        # ניסיון לכל קודקוד כנקודת התחלה
        for start_node in self.nodes:
            cycle, avg = self._find_max_avg_cycle_from_node(start_node)
            if avg > max_avg:
                max_avg = avg
                max_cycle = cycle

        return max_cycle, max_avg

    def _find_max_avg_cycle_from_node(self, start_node):
        distances = {node: float('-inf') for node in self.nodes}
        parents = {node: None for node in self.nodes}
        distances[start_node] = 0

        pq = [(-0, start_node)]
        visited = set()

        while pq:
            neg_dist, current = heapq.heappop(pq)
            dist = -neg_dist

            if current in visited:
                continue

            visited.add(current)

            for neighbor, weight in self.graph[current]:
                if distances[neighbor] < dist + weight:
                    distances[neighbor] = dist + weight
                    parents[neighbor] = current
                    if neighbor not in visited:
                        heapq.heappush(pq, (-(dist + weight), neighbor))

        best_cycle = None
        best_avg = float('-inf')

        for neighbor, weight in self.graph[start_node]:
            if neighbor == start_node:
                if weight > best_avg:
                    best_avg = weight
                    best_cycle = [start_node, start_node]
            elif neighbor in visited:
                path = self._reconstruct_path(parents, start_node, neighbor)
                if path:
                    path.append(start_node)

                    cycle_weight = 0
                    for i in range(len(path) - 1):
                        for next_node, edge_weight in self.graph[path[i]]:
                            if next_node == path[i + 1]:
                                cycle_weight += edge_weight
                                break

                    avg_weight = cycle_weight / (len(path) - 1)
                    if avg_weight > best_avg:
                        best_avg = avg_weight
                        best_cycle = path

        return best_cycle, best_avg

    def _reconstruct_path(self, parents, start, end):
        path = [end]
        current = parents[end]

        if current is None:
            return []

        while current != start:
            path.append(current)
            current = parents[current]
            if current in path:
                print("Cycle detected in path reconstruction")
                return []
            if current is None:
                return []

        path.append(start)
        path.reverse()
        return path


if __name__ == "__main__":
    g = EnvyGraph()

    # הוספת קודקודים עם ערכי החפצים שלהם
    # נניח שכל קודקוד הוא אדם, והחפץ שלו הוא באותו שם (לדוגמה: A הוא אדם והחפץ שלו הוא A)
    # -------------------- הדוגמה כאן היא הדוגמה מההרצאה --------------------
    print("Example 1:")
    g.add_node('A', 'A', 50, {'B': 30})
    g.add_node('B', 'B', 40, {'A': 90})

    # בניית צלעות הקנאה אוטומטית
    g.build_envy_edges()

    # הצגת משקלי הקנאה שחושבו
    print("משקלי הקנאה שחושבו:")
    for source in g.nodes:
        for target, weight in g.graph[source]:
            print(f"{source} -> {target}: {weight}")

    # מציאת המעגל עם המשקל הממוצע הגבוה ביותר
    cycle, avg_weight = g.find_max_avg_weight_cycle()
    print(f"Cycle found: {cycle}, Avg Weight: {avg_weight}")

    # -------------------- דוגמה בה המעגל מ-ק לעצמו מקסימלי --------------------
    print(f"\nExample 2 with weight 0:")
    g.add_node('A', 'A', 50, {'B': 10})
    g.add_node('B', 'B', 40, {'A': 30})

    # בניית צלעות הקנאה אוטומטית
    g.build_envy_edges()

    # הוספת צלעות מקודקוד לעצמו במשקל 0
    g.add_envy_edge('A', 'A', 0)
    g.add_envy_edge('B', 'B', 0)

    # הצגת משקלי הקנאה שחושבו
    print("משקלי הקנאה שחושבו:")
    for source in g.nodes:
        for target, weight in g.graph[source]:
            print(f"{source} -> {target}: {weight}")

    # מציאת המעגל עם המשקל הממוצע הגבוה ביותר
    cycle, avg_weight = g.find_max_avg_weight_cycle()
    print(f"Cycle found: {cycle}, Avg Weight: {avg_weight}")
